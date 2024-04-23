#!/usr/bin/python3

import argparse
import sys
import os
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import S, BtoA, AtoB
from datasets import ImageDataset

#print("Time of operation...")
if not os.path.exists('Output/S-color0.5/'):
    os.makedirs('Output/S-color0.5/')
output_file = open("./Output/S-color0.5/runtime.txt", "w")
start_time = time.time()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/S-color0.5/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default='true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='Output/S-color0.5/model/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='Output/S-color0.5/model/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--generator_E1', type=str, default='Output/S-color0.5/model/netG_E1.pth', help='E generator checkpoint file')
parser.add_argument('--generator_E2', type=str, default='Output/S-color0.5/model/netG_E2.pth', help='E generator checkpoint file')
opt = parser.parse_args()
print(opt) 

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = AtoB(opt.input_nc, opt.output_nc)
netG_B2A = BtoA(opt.output_nc, opt.input_nc)
netG_E1 = S(opt.output_nc, opt.input_nc)
netG_E2 = S(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netG_E1.cuda()
    netG_E2.cuda()


# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B),strict=False)
netG_B2A.load_state_dict(torch.load(opt.generator_B2A),strict=False)
netG_E1.load_state_dict(torch.load(opt.generator_E1),strict=False)
netG_E2.load_state_dict(torch.load(opt.generator_E2),strict=False)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()
netG_E1.eval()
netG_E2.eval()

#print('GENTOT')

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Resize(size=(256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('Output/S-color0.5/result/img_a11'):
    os.makedirs('Output/S-color0.5/result/img_a11')
if not os.path.exists('Output/S-color0.5/result/img_b11'):
    os.makedirs('Output/S-color0.5/result/img_b11')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*((netG_A2B(real_A)).data + 1.0)
    fake_A = 0.5*((netG_B2A(real_B)).data + 1.0)

    # Save image files
    save_image(fake_A, 'Output/S-color0.5/result/img_a11/%04d.png' % (i + 1))
    save_image(fake_B, 'Output/S-color0.5/result/img_b11/%04d.png' % (i + 1))

    #print('save after')
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
end_time = time.time()
print(f"Time of operation in {end_time-start_time:.4f} seconds")


total_time = end_time - start_time  
avg_img_time = total_time / len(dataloader) 

output_file.write(f"\nTotal time of operation: {total_time:.4f} seconds\n")
output_file.write(f"Average time of operation per image: {avg_img_time:.4f} seconds\n")
output_file.close() 