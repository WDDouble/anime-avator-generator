import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms

class EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * np.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)
        
    def forward(self, input):
        return self.conv(input)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

  
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel1, pad1, kernel2, pad2, pixel_norm=True):
        super().__init__()

        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.stride1 = 1
        self.stride2 = 1
        self.pad1 = pad1
        self.pad2 = pad2

        if pixel_norm:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel1, self.stride1, self.pad1),
                                      PixelNorm(),
                                      nn.LeakyReLU(0.2),
                                      EqualConv2d(out_channel, out_channel, self.kernel2, self.stride2, self.pad2),
                                      PixelNorm(),
                                      nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel1, self.stride1, self.pad1),
                                      nn.LeakyReLU(0.2),
                                      EqualConv2d(out_channel, out_channel, self.kernel2, self.stride2, self.pad2),
                                      nn.LeakyReLU(0.2))
    def forward(self, input):
        out = self.conv(input)
        return out




class Generator(nn.Module):
    def __init__(self, code_dim=512):
        super().__init__()
        self.code_norm = PixelNorm()
        self.progression = nn.ModuleList([ConvBlock(512, 512, 4, 3, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 256, 3, 1, 3, 1),
                                          ConvBlock(256, 128, 3, 1, 3, 1)])
        self.to_rgb = nn.ModuleList([nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(256, 3, 1),
                                     nn.Conv2d(128, 3, 1),])
        
    def forward(self, input, expand=0, alpha=-1):
        out = self.code_norm(input)
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and expand > 0:
                upsample = F.interpolate(out, scale_factor=2)
                out = conv(upsample)
            else:
                out = conv(out)
                
            if i == expand:
                out = to_rgb(out)
                
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break
            
        return out


class Distriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(256, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(513, 512, 3, 1, 4, 0, pixel_norm=False),])
        self.from_rgb = nn.ModuleList([nn.Conv2d(3, 128, 1),
                                       nn.Conv2d(3, 256, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),])
        self.n_layer = len(self.progression)
        self.linear = nn.Linear(512, 1)
    
    def forward(self, input, expand=0, alpha=-1):
        for i in range(expand, -1, -1):
            index = self.n_layer - i - 1
            if i == expand:
                out = self.from_rgb[index](input)
            if i == 0:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)
            out = self.progression[index](out)
            
            if i > 0:
                out = F.avg_pool2d(out, 2)
                if i == expand and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out
                    
        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out