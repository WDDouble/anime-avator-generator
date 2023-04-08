import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms

n_generator_feature = 128      # 生成器feature map数
n_discriminator_feature = 128 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, n_generator_feature * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LayerNorm([n_generator_feature * 8, 4, 4]),
            nn.ReLU(True),       # (n_generator_feature * 8) × 4 × 4        (1-1)*1+1*(4-1)+0+1 = 4
            nn.ConvTranspose2d(n_generator_feature * 8, n_generator_feature * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([n_generator_feature * 4, 8, 8]),
            nn.ReLU(True),      # (n_generator_feature * 4) × 8 × 8     (4-1)*2-2*1+1*(4-1)+0+1 = 8
            nn.ConvTranspose2d(n_generator_feature * 4, n_generator_feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([n_generator_feature * 2, 16, 16]),
            nn.ReLU(True),  # (n_generator_feature * 2) × 16 × 16
            nn.ConvTranspose2d(n_generator_feature * 2, n_generator_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([n_generator_feature, 32, 32]),
            nn.ReLU(True),      # (n_generator_feature) × 32 × 32
            nn.ConvTranspose2d(n_generator_feature, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()       # 3 * 96 * 96
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        ndf = n_discriminator_feature
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入3*96*96即生成器生成的图片
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),  # 卷积，下采样，也是编码的过程
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 ndf x 32 x32

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 正好将feature map图片大小缩小一半
            nn.LayerNorm([n_discriminator_feature * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([n_discriminator_feature * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([n_discriminator_feature * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 最后编码成为一个维度为1的向量
            # nn.Sigmoid()  # 最后用Sigmoid作为分类，使得判别器成为一个判断二分类问题的模型，实际上判别器也是做一个二分类任务，判断是否为原图输出0或1
        )

    def forward(self, input):
        return self.main(input).view(-1)
    
def gradient_penalty(D, real, fake, device, gp_weight=10):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad = True

    # Get the discriminator output for the interpolated images
    d_interpolated = D(interpolated)

    # Calculate gradients w.r.t the interpolated images
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(d_interpolated.size(), device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Calculate the gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gp = gp_weight * ((gradients_norm - 1) ** 2).mean()

    return gp

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)
