import numpy as np
import torch.nn as nn
import torch

import torch.optim as optim

from lib.generators import PatchGenerator
from lib.discriminator import Discriminator

class QGAN:
    def __init__(
        self,
        image_size,
        gen_count,
        gen_arch,
        input_state,
        noise_dim,
        batch_size,
        pnr,
        lossy,
        remote_token=None,
        use_clements=False,
        sim = False
    ):
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.remote_token = remote_token
        self.G = PatchGenerator(
            image_size,
            gen_count,
            gen_arch,
            input_state,
            pnr,
            lossy,
            remote_token,
            use_clements,
            sim = sim
        )
        self.D = Discriminator(image_size)
        self.batch_size = batch_size
        self.fake_data = []

    def get_G_loss(self, fake_data=None):

        if fake_data is None:
            fake = self.G.forward()
        else:
            fake = fake_data

        pred_fake = self.D(fake)
        G_loss = -torch.mean(torch.log(pred_fake + 1e-8))

        return G_loss

        
    def fit(self, dataloader, lrD, lrG, opt_params, silent=False, callback=None):
        opt_iter_num = opt_params["opt_iter_num"]

        params_prog = []
        fake_progress = []

        criterion = nn.BCELoss()
        D_params = self.D.parameters()
        G_params = self.G.init_params()

        real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float)
        fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float)

        fixed_noise = np.random.normal(0, 2 * np.pi, (self.batch_size, self.noise_dim))
        fake_progress.extend(self.G.generate(fixed_noise))

        optD = optim.SGD(D_params, lr=lrD)
        optG = optim.SGD(G_params, lr=lrG)

        G_loss_prog = []
        D_loss_prog = []

        for i, (data, _) in enumerate(dataloader):
            real_data = data.reshape(-1, self.image_size * self.image_size)
            noise = np.random.normal(0, 2 * np.pi, (self.batch_size, self.noise_dim))
            fake_data = self.G.generate(noise)
            self.fake_data = fake_data

            # discriminator training
            self.D.zero_grad()
            outD_real = self.D(real_data).view(-1)
            outD_fake = self.D(fake_data).view(-1)

            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)

            errD_real.backward()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optD.step()
            D_loss = errD.detach().item()

            # generator training
            G_loss_function = self.get_G_loss
            self.G.zero_grad()
            G_loss_function.backward()
            optG.step()
            G_loss = G_loss_function() 

            # log and display results
            D_loss_prog.append(D_loss)
            G_loss_prog.append(G_loss)
            params_prog.append(G_params)

            if not silent:
                print("it", i)
                print("D_loss", D_loss)
                print("G_loss", G_loss)

            fake_samples = None
            step_interval = max(1, opt_iter_num // 100)
            if i % step_interval == 0:
                fake_samples = self.G.forward(fixed_noise)
                fake_progress.extend(fake_samples)

            if callback is not None:
                callback(
                    i, D_loss, G_loss, G_params, self.D.state_dict(), fake_samples, optG
                )

        return D_loss_prog, G_loss_prog, params_prog, fake_progress
