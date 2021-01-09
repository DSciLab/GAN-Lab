from torch import autograd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from mlutils import Trainer
from networks.gan import vanilla


class VanillaGANTrainer(Trainer):
    def __init__(self, opt):
        super().__init__(opt)
        G = vanilla.Generator(opt)
        D = vanilla.Discriminator(opt)
        self.G = self.to_gpu(G)
        self.D = self.to_gpu(D)
        self.optimizerG = Adam(G.parameters(), lr=opt.lr)
        self.optimizerD = Adam(D.parameters(), lr=opt.lr)
        self.bce_loss = F.binary_cross_entropy

    def update_label(self, batch_size):
        self.valid = autograd.Variable(torch.Tensor(batch_size, 1).fill_(1.0), 
                        requires_grad=False)
        self.fake = autograd.Variable(torch.Tensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)
        
        self.valid = self.to_gpu(self.valid)
        self.fake = self.to_gpu(self.fake)

    def train_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)
        images = torch.squeeze(images)
        batch_size = images.size(0)
        self.update_label(batch_size)

        z = self.G.sample_z(batch_size)
        z = self.to_gpu(z)
        fake = self.G(z)

        # train generator
        self.optimizerG.zero_grad()
        g_loss = self.bce_loss(self.D(fake), self.valid)
        g_loss.backward()
        self.optimizerG.step()

        # train discriminator
        self.optimizerD.zero_grad()
        d_fake_loss = self.bce_loss(self.D(fake.detach()), self.fake)
        d_real_loss = self.bce_loss(self.D(images), self.valid)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizerD.step()

        self.dashboard.add_trace_dict({'g_loss': g_loss.detach(),
                                       'd_loss': d_loss.detach()}, self.step)
        self.dashboard.add_image_dict({'fake_image': fake,
                                       'real_image': images})
        loss = d_loss.detach() + g_loss.detach()
        return loss.detach(), None, None

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)
        images = torch.squeeze(images)
        batch_size = images.size(0)
        self.update_label(batch_size)

        z = self.G.sample_z(batch_size)
        z = self.to_gpu(z)
        fake = self.G(z)

        # train generator
        g_loss = self.bce_loss(self.D(fake), self.valid)

        # train discriminator
        d_fake_loss = self.bce_loss(self.D(fake.detach()), self.fake)
        d_real_loss = self.bce_loss(self.D(images), self.valid)
        d_loss = d_real_loss + d_fake_loss

        self.dashboard.add_image_dict({'fake_image': fake,
                                       'real_image': images})
        loss = d_loss.detach() + g_loss.detach()
        return loss.detach(), None, None

    def infer(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        mu, logvar, gen_images = self.model(images)
        loss = self.loss_fn(gen_images, images, mu, logvar)

        # self.dashboard.add_image_dict({'gen_image': gen_images,
        #                                 'image': images})
        return loss.detach(), gen_images, images

    def infer_image(self, image):
        image = self.to_gpu(image)
        _, _, gen_image = self.model(image)
        self.dashboard.add_image_dict(
                            {'gen_image': gen_image,
                             'image': image})
        return gen_image

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass
