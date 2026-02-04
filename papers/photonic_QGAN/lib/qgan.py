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

        
    def fit(self, dataloader, lrD, lrG, opt_params, silent=False, callback=None):
        opt_iter_num = opt_params["opt_iter_num"]

        fake_progress = []

        criterion = nn.BCEWithLogitsLoss()
        D_params = self.D.parameters()
        G_params = self.G.parameters()

        device = next(self.D.parameters()).device
        fixed_noise = torch.normal(0, 2 * torch.pi, (self.batch_size, self.noise_dim), device=device)
        with torch.no_grad():
            fake_progress.append(self.G(fixed_noise).detach().cpu())


        optD = optim.Adam(self.D.parameters(), lr=lrD, betas=(0.5, 0.999))
        optG = optim.Adam(self.G.parameters(), lr=lrG, betas=(0.5, 0.999))

        G_loss_prog = []
        D_loss_prog = []

        for i, (data, _) in enumerate(dataloader):
            real_data = data.reshape(data.size(0), -1)

            real_data = real_data.to(device)

            B = real_data.size(0)
            noise = torch.normal(0, 2 * torch.pi, (B, self.noise_dim), device=device)

            real_labels = torch.ones(B, device=device)
            fake_labels = torch.zeros(B, device=device)

            fake_data = self.G(noise)

            # discriminator training
            self.D.zero_grad()
            outD_real = self.D(real_data).view(-1)                 # logits
            outD_fake = self.D(fake_data.detach()).view(-1)        # logits, detached

            errD = criterion(outD_real, real_labels) + criterion(outD_fake, fake_labels)
            errD.backward()
            optD.step()
            
            D_loss = errD.detach().item()

            # freeze discriminator
            for p in self.D.parameters():
                p.requires_grad_(False)

            # generator training
            self.G.zero_grad()
            fake_data = self.G(noise)
            outD_fake_for_G = self.D(fake_data).view(-1)           # logits, NOT detached
            G_loss = criterion(outD_fake_for_G, real_labels)
            G_loss.backward()
            optG.step()

            # unfreeze
            for p in self.D.parameters():
                p.requires_grad_(True)
            
            G_loss_val = G_loss.detach().item()
            
            # log and display results
            D_loss_prog.append(D_loss)
            G_loss_prog.append(G_loss_val)

            if not silent:
                print("it", i)
                print("D_loss", D_loss)
                print("G_loss", G_loss_val)

            fake_samples = None
            step_interval = max(1, opt_iter_num // 100)
            if i % step_interval == 0:
                with torch.no_grad():
                    fake_samples = self.G(fixed_noise.to(device)).detach().cpu()
                fake_progress.append(fake_samples)

            if callback is not None:
                callback(
                    i, D_loss, G_loss_val, self.G.state_dict(), self.D.state_dict(), fake_samples, optG
                )

        final_G_params = self.G.state_dict()

        return D_loss_prog, G_loss_prog, final_G_params, fake_progress
