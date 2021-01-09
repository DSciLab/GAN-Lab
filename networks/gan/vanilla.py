import torch
from torch import nn
from torch import autograd
import numpy as np


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.image_shape = (opt.image_size, opt.image_size)
        self.latent_dim = opt.z_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )
    
    def sample_z(self, batch_size):
        z = autograd.Variable(torch.randn(batch_size, self.latent_dim))
        return z

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        image_shape = (opt.image_size, opt.image_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

