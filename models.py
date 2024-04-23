
# -*- coding: UTF-8 -*-
# -*- coding: gbk -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)  


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),  
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.pa(x)
        return x * y + x
        #return out




class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y + x

class FAM(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(FAM, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class TFAM(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(TFAM, self).__init__()
        modules = [FAM(conv, dim, kernel_size) for _ in range(blocks)]
        

        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
        

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res



class De_remove(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(De_remove, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = TFAM(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = TFAM(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = TFAM(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        

        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  
        # w.size() = ([1, 3, 64, 1, 1])

        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        

        out = self.palayer(out)
        x = self.post(out)
        return x + x1

##########################################################################


class Re_pretict(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(Re_pretict, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_9(x1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        #x1 = self.self_attn(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x + out
        return out2
##########################################################################


class De_predict(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(De_predict, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        #layers = []
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.InstanceNorm2d(features),nn.LeakyReLU(0.2, inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.InstanceNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_9(x1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x - out
        return out2
##########################################################################
##########################################################################
##########################################################################
##########################################################################

class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x
        
##########################################################################
#CAB
##########################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)
        act = nn.LeakyReLU(0.2)
        self.act = act
        
        self.gcnet = ContextBlock(in_features)
    def forward(self, x):
        res = self.conv_block(x)
        res = self.act(self.gcnet(res))
        res += x
        return res
        #return x + self.conv_block(x)

##########################################################################

##########################################################################
class AtoB(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(AtoB, self).__init__()
        f1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True),
                 nn.Dropout2d(0.2)]
        self.f1 = nn.Sequential(*f1)
        
        in_features = 64
        out_features = in_features*2
        
        f2 = [nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2),
                      nn.MaxPool2d(2)]
        
        in_features = out_features
        out_features = in_features*2
        self.f2 = nn.Sequential(*f2)
        
        f3 = [nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2),
                      nn.MaxPool2d(2)]
        in_features = out_features
        out_features = in_features*2
        self.f3 = nn.Sequential(*f3)
        
        for _ in range(n_residual_blocks):
            f4 = [ResidualBlock(in_features)]
        self.f4 = nn.Sequential(*f4)
        out_features = in_features//2   
        
        f5 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2)]
        self.f5 = nn.Sequential(*f5)
        in_features = out_features
        out_features = in_features//2
        
        f6 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2)]
        self.f6 = nn.Sequential(*f6)
        in_features = out_features
        out_features = in_features//2
        
        f7 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]
        self.f7 = nn.Sequential(*f7)
        self.Dep = De_predict(channels=3)
        self.Der = De_remove(gps=3, blocks=3)
        
        f8 = [nn.Conv2d(6, output_nc, 3, stride=1, padding=1),
                  nn.Tanh()]
        self.f8 = nn.Sequential(*f8)
        
    def forward(self, x):
        xa1 = self.Dep(x)
        xa2 = self.Der(x)
        result = torch.cat([xa1, xa2], dim=1)
        result = self.f8(result)
        x1 = self.f1(result)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        x4 = x4 + x3
        x5 = self.f5(x4)
        x5 = x5 + x2
        x6 = self.f6(x5)
        x6 = x6 + x1
        x7 = self.f7(x6)
        return x7 
##########################################################################
##########################################################################
##########################################################################
class BtoA(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(BtoA, self).__init__()
        f1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True),
                 nn.Dropout2d(0.2)]
        self.f1 = nn.Sequential(*f1)
        
        in_features = 64
        out_features = in_features*2
        
        f2 = [nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2),
                      nn.MaxPool2d(2)]
        
        in_features = out_features
        out_features = in_features*2
        self.f2 = nn.Sequential(*f2)
        
        f3 = [nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2),
                      nn.MaxPool2d(2)]
        in_features = out_features
        out_features = in_features*2
        self.f3 = nn.Sequential(*f3)
        
        for _ in range(n_residual_blocks):
            f4 = [ResidualBlock(in_features)]
        self.f4 = nn.Sequential(*f4)
        out_features = in_features//2   
        
        f5 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2)]
        self.f5 = nn.Sequential(*f5)
        in_features = out_features
        out_features = in_features//2
        
        f6 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2)]
        self.f6 = nn.Sequential(*f6)
        in_features = out_features
        out_features = in_features//2
        
        f7 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]
        self.f7 = nn.Sequential(*f7)
        self.AD = Re_pretict(channels=3)
    def forward(self, x):
        xa1 = self.AD(x)
        x1 = self.f1(x)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        x4 = x4 + x3
        x5 = self.f5(x4)
        x5 = x5 + x2
        x6 = self.f6(x5)
        x6 = x6 + x1
        x7 = self.f7(x6)
        x7 = x7 + xa1
        return x7 
##########################################################################
##########################################################################
##########################################################################
class S(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(S, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
##########################################################################
##########################################################################
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # 4*Conv layers coupled with leaky-relu & instance norm.
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Final layer.
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
##########################################################################
##########################################################################
##########################################################################
if __name__ == "__main__":
    net1 = Generator(input_nc=3, output_nc=3)
    print(net1)

