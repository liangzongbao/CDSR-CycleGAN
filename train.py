#!/usr/bin/python3

import argparse
import itertools
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn


from models import S, BtoA, AtoB
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from utils import VGGNet

# python -m visdom.server
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1,
                    help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/S-color0.5',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3,
                    help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default='true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0,
                    help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

# Networks
netG_A2B = AtoB(opt.input_nc, opt.output_nc)
netG_B2A = BtoA(opt.output_nc, opt.input_nc)
netG_E1 = S(opt.output_nc, opt.output_nc)
netG_E2 = S(opt.input_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netG_E1.cuda()
    netG_E2.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netG_E1.apply(weights_init_normal)
netG_E2.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# pretrained VGG19 module set in evaluation mode for feature extraction
vgg = VGGNet().cuda().eval()


def perceptual_loss(x, y):
    c = nn.MSELoss()

    rx = netG_B2A(netG_A2B(x))
    ry = netG_A2B(netG_B2A(y))

    fx1, fx2 = vgg(x)
    fy1, fy2 = vgg(y)

    frx1, frx2 = vgg(rx)
    fry1, fry2 = vgg(ry)

    m1 = c(fx1, frx1)
    m2 = c(fx2, frx2)

    m3 = c(fy1, fry1)
    m4 = c(fy2, fry2)

    loss = (m1 + m2 + m3 + m4)

    return loss


# Lossess
criterion_GAN = torch.nn.MSELoss()  # Adversarial Loss
criterion_cycle = torch.nn.L1Loss()  # Cyclic consistency loss
criterion_identity = torch.nn.L1Loss()  # identity loss
criterion_fGT = torch.nn.L1Loss()  # Pseudo-similarity loss

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(), netG_E2.parameters(), netG_E2.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0),
                       requires_grad=False)  # real
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0),
                       requires_grad=False)  # fake

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
E_A_buffer = ReplayBuffer()
E_B_buffer = ReplayBuffer()

# Dataset loader with data augmentations
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

if not os.path.exists('Output/S-color0.5/model'):
    os.makedirs('Output/S-color0.5/model')
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A'])).cuda()
        real_B = Variable(input_B.copy_(batch['B'])).cuda()

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B1 = criterion_GAN(pred_fake, target_real)
        fake_BE = netG_E1(fake_B)
        pred_fake2 = netD_B(fake_BE)
        loss_GAN_A2B2 = criterion_GAN(pred_fake2, target_real)
        loss_GAN_A2B = loss_GAN_A2B1 + loss_GAN_A2B2

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A1 = criterion_GAN(pred_fake, target_real)
        fake_AE = netG_E2(fake_A)
        pred_fake2 = netD_B(fake_AE)
        loss_GAN_B2A2 = criterion_GAN(pred_fake2, target_real)
        loss_GAN_B2A = loss_GAN_B2A1 + loss_GAN_B2A2

        # Identity loss
        loss_id_B2A = criterion_identity(netG_B2A(real_A), real_A) * 5.0
        loss_id_A2B = criterion_identity(netG_A2B(real_B), real_B) * 5.0

        # loss_identity = (loss_id_A + loss_id_B) / 2

        # Cycle loss
        recovered_A = netG_B2A(fake_B)

        recovered_B = netG_A2B(fake_A)

        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        loss_cycle_1 = (loss_cycle_ABA + loss_cycle_BAB)
        # Cycle loss1
        #E_A = netG_E1(fake_B)
        #E_B = netG_E2(fake_A)
        recovered_A2 = netG_B2A(fake_BE)
        recovered_B2 = netG_A2B(fake_AE)

        loss_cycle_ABEA = criterion_cycle(recovered_A2, real_A) * 10.0
        loss_cycle_BAEB = criterion_cycle(recovered_B2, real_B) * 10.0
        loss_cycle_2 = (loss_cycle_ABEA + loss_cycle_BAEB)
        # Perceptual loss
        loss_perceptual = perceptual_loss(real_A, real_B)
        # ps loss
        loss_fGT = (criterion_fGT(fake_B, fake_BE) + criterion_fGT(fake_A, fake_AE)) * 0.5

        # Total loss
        loss_G = loss_perceptual * 0.7 + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_1 + loss_cycle_2 + loss_id_B2A + loss_id_A2B + loss_fGT
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake1 = criterion_GAN(pred_fake, target_fake)
        E_B = E_B_buffer.push_and_pop(fake_AE)
        pred_fake1 = netD_A(E_B.detach())
        loss_D_fake2 = criterion_GAN(pred_fake1, target_fake)
        loss_D_fake = loss_D_fake1 + loss_D_fake2
        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake1 = criterion_GAN(pred_fake, target_fake)
        E_A = E_A_buffer.push_and_pop(fake_BE)
        pred_fake1 = netD_B(E_A.detach())
        loss_D_fake2 = criterion_GAN(pred_fake1, target_fake)
        loss_D_fake = loss_D_fake1 + loss_D_fake2
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        # logged using visdom
        logger.log({'L_G': loss_G, 'L_G_perceptual': loss_perceptual, 'L_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'L_G_cycle1': (loss_cycle_1), 'L_G_cycle2': (loss_cycle_2), 'L_fGT': (loss_fGT),
                    'L_G_identity': (loss_id_B2A + loss_id_A2B), 'L_D': (loss_D_A + loss_D_B)},
                   images={'real_A': real_A,  'fake_B': fake_B, 'recovered_A': recovered_A, 'fake_BE': fake_BE,  'recovered_A2': recovered_A2,
                           'real_B': real_B,  'fake_A': fake_A, 'recovered_B': recovered_B, 'fake_AE': fake_AE,  'recovered_B2': recovered_B2})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'Output/S-color0.5/model/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'Output/S-color0.5/model/netG_B2A.pth')
    torch.save(netG_E1.state_dict(), 'Output/S-color0.5/model/netG_E1.pth')
    torch.save(netG_E2.state_dict(), 'Output/S-color0.5/model/netG_E2.pth')
    torch.save(netD_A.state_dict(), 'Output/S-color0.5/model/netD_A.pth')
    torch.save(netD_B.state_dict(), 'Output/S-color0.5/model/netD_B.pth')
###################################




