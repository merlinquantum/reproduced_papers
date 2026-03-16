import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lib.discriminator import Discriminator
from lib.generators import PatchGenerator
from utils.spsa import SPSA


def bernoulli_delta(n_params, p=0.5):
    delta_k = np.random.binomial(1, p, n_params)
    delta_k[delta_k == 0] = -1
    return delta_k


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
        sim=False,
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
            sim=sim,
        )
        self.D = Discriminator(image_size)

        self.batch_size = batch_size
        self.fake_data = []
        self.G_loss = None

    def get_G_loss_diff(self, params, diff):
        self.G.update_var_params(params + diff)
        iteration_list_pos = self.G.get_iteration_list()

        self.G.update_var_params(params - diff)
        iteration_list_neg = self.G.get_iteration_list()

        iteration_list = iteration_list_pos + iteration_list_neg
        result = self.G.generate(it_list=iteration_list)

        pos = result[: (len(result) // 2)]
        neg = result[(len(result) // 2) :]

        loss_pos = self.get_G_loss(fake_data=pos)
        loss_neg = self.get_G_loss(fake_data=neg)

        return loss_pos, loss_neg

    def get_G_loss(self, params=None, fake_data=None):
        if params is not None:
            self.G.update_var_params(params)
        elif fake_data is None:
            return self.G_loss

        if fake_data is None:
            fake = self.G.generate()
        else:
            fake = fake_data

        pred_fake = self.D(fake).detach().numpy()
        G_loss = -np.mean(np.log(pred_fake))
        self.G_loss = G_loss

        return G_loss

    def grad_G(self, params, c):
        delta_k = bernoulli_delta(len(params))
        loss_pos, loss_neg = self.get_G_loss_diff(params, c * delta_k)

        grads = []
        for i in range(len(params)):
            grads.append((loss_pos - loss_neg) / (2 * c * delta_k[i]))

        self.G.update_var_params(params)
        return np.array(grads)

    # TODO: where the PyTorch otpimizer (Adam, SGD...) should be used
    def fit(self, dataloader, lrD, opt_params, silent=False, callback=None):
        spsa_iter_num = opt_params["spsa_iter_num"]
        opt_iter_num = opt_params["opt_iter_num"]

        params_prog = []
        fake_progress = []
        spsa_step_duration = max(1, spsa_iter_num // opt_iter_num)

        criterion = nn.BCELoss()
        D_params = self.D.parameters()
        G_params = self.G.init_params()

        real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float)
        fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float)

        fixed_noise = np.random.normal(0, 2 * np.pi, (self.batch_size, self.noise_dim))
        fake_progress.extend(self.G.generate(fixed_noise))

        optD = optim.SGD(D_params, lr=lrD)
        optG = SPSA(G_params, self.grad_G, spsa_iter_num)

        # ability to specify optimisation params for interrupted optimizations
        if "a" in opt_params:
            optG.a = opt_params["a"]
        if "k" in opt_params:
            optG.k = opt_params["k"]

        # raise exception if a is not in desired range
        if self.remote_token is None and (optG.a > 100 or optG.a < 10):
            raise Exception("SPSA: a value out of range. Reinitialize.")

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
            G_params = optG.step(spsa_step_duration)
            self.G.update_var_params(G_params)
            G_loss = self.G_loss

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
                fake_samples = self.G.generate(fixed_noise)
                fake_progress.extend(fake_samples)

            if callback is not None:
                callback(
                    i, D_loss, G_loss, G_params, self.D.state_dict(), fake_samples, optG
                )

        return D_loss_prog, G_loss_prog, params_prog, fake_progress
