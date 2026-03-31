import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lib.discriminator import Discriminator
from lib.generators import ClassicalGenerator, PatchGenerator
from lib.spsa import SPSA, bernoulli_delta
from skimage.metrics import structural_similarity as ssim


def get_metrics(real, fake):
    count = len(real)
    if count < 2:
        return 0.0, 0.0

    diversity = 0.0
    similarity = 0.0
    for i in range(count):
        for j in range(count):
            similarity += ssim(real[i], fake[j], data_range=1.0)
        for j in range(i + 1, count):
            diversity += ssim(fake[i], fake[j], data_range=1.0)

    similarity /= count * count
    denom = count * (count - 1) / 2
    diversity /= denom

    return similarity, 1 - diversity


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
        generator_type="photonic",
    ):
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.remote_token = remote_token
        generator_key = str(generator_type).strip().lower()
        if generator_key == "classical":
            self.G = ClassicalGenerator(
                noise_dim=noise_dim,
                image_size=image_size,
            )
        else:
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

    # ------------------------------------------------------------------
    # SPSA helpers
    # ------------------------------------------------------------------

    def _get_G_params_flat(self):
        """Return all trainable generator parameters as a flat numpy array."""
        parts = [p.detach().cpu().numpy().reshape(-1) for p in self.G.parameters()]
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    def _set_G_params_flat(self, flat_params):
        """Write a flat numpy array back into the generator's trainable parameters."""
        offset = 0
        with torch.no_grad():
            for p in self.G.parameters():
                n = p.numel()
                new_val = torch.tensor(
                    flat_params[offset : offset + n].reshape(p.shape),
                    dtype=p.dtype,
                    device=p.device,
                )
                p.copy_(new_val)
                offset += n

    def _compute_G_loss_no_grad(self, noise, gen_labels, criterion):
        """Forward pass through G and D without building a computation graph."""
        with torch.no_grad():
            fake = self.G(noise)
            return criterion(self.D(fake).view(-1), gen_labels).item()

    def fit(
        self,
        dataloader,
        lrD,
        lrG,
        opt_iter_num,
        train_params=None,
        silent=False,
        callback=None,
    ):
        if isinstance(opt_iter_num, dict):
            payload = dict(opt_iter_num)
            opt_iter_num = int(payload.get("opt_iter_num", 0))
            if train_params is None:
                train_params = {}
            train_params.update(
                {k: v for k, v in payload.items() if k != "opt_iter_num"}
            )
        if train_params is None:
            train_params = {}

        if opt_iter_num <= 0:
            raise ValueError("opt_iter_num must be positive.")

        optimizer_type = str(train_params.get("optimizer", "adam")).strip().lower()
        d_optimizer_type = str(train_params.get("d_optimizer", "adam")).strip().lower()

        adam_beta1 = float(train_params.get("adam_beta1", 0.5))
        adam_beta2 = float(train_params.get("adam_beta2", 0.999))
        real_label_value = float(train_params.get("real_label", 0.9))
        fake_label_value = float(train_params.get("fake_label", 0.0))
        gen_target_value = float(train_params.get("gen_target", real_label_value))
        d_steps = int(train_params.get("d_steps", 1))
        g_steps = int(train_params.get("g_steps", 1))
        if d_steps <= 0 or g_steps <= 0:
            raise ValueError("d_steps and g_steps must be positive.")

        fake_progress = []

        criterion = nn.BCEWithLogitsLoss()

        device = next(self.D.parameters()).device
        fixed_noise = torch.normal(
            0, 2 * torch.pi, (self.batch_size, self.noise_dim), device=device
        )
        with torch.no_grad():
            fake_progress.append(self.G(fixed_noise).detach().cpu())

        if d_optimizer_type == "sgd":
            optD = optim.SGD(self.D.parameters(), lr=lrD)
        else:
            optD = optim.Adam(
                self.D.parameters(), lr=lrD, betas=(adam_beta1, adam_beta2)
            )

        # ---- Generator optimizer setup ----
        if optimizer_type == "spsa":
            spsa_iter_num = int(train_params.get("spsa_iter_num", 10500))
            spsa_step_duration = max(1, spsa_iter_num // opt_iter_num)

            # Mutable context so grad_G can read the current iteration's tensors.
            _spsa_ctx: dict = {
                "noise": fixed_noise,
                "gen_labels": None,
                "criterion": criterion,
            }

            fixed_gen_labels = torch.full(
                (self.batch_size,), gen_target_value, device=device
            )
            _spsa_ctx["gen_labels"] = fixed_gen_labels

            def _grad_G(params, c):
                delta = bernoulli_delta(len(params))
                self._set_G_params_flat(params + c * delta)
                loss_pos = self._compute_G_loss_no_grad(
                    _spsa_ctx["noise"], _spsa_ctx["gen_labels"], _spsa_ctx["criterion"]
                )
                self._set_G_params_flat(params - c * delta)
                loss_neg = self._compute_G_loss_no_grad(
                    _spsa_ctx["noise"], _spsa_ctx["gen_labels"], _spsa_ctx["criterion"]
                )
                self._set_G_params_flat(params)
                return (loss_pos - loss_neg) / (2 * c * delta)

            init_params = self._get_G_params_flat()
            optG = SPSA(init_params, _grad_G, spsa_iter_num)

            # Override a/k if supplied (matching original behaviour).
            if "spsa_a" in train_params:
                optG.a = float(train_params["spsa_a"])
            if "spsa_k" in train_params:
                optG.k = int(train_params["spsa_k"])
        else:
            optG = optim.Adam(
                self.G.parameters(), lr=lrG, betas=(adam_beta1, adam_beta2)
            )

        G_loss_prog = []
        D_loss_prog = []
        ssim_prog = []

        for i, (data, _) in enumerate(dataloader):
            real_data = data.reshape(data.size(0), -1)

            real_data = real_data.to(device)

            B = real_data.size(0)
            real_labels = torch.full((B,), real_label_value, device=device)
            fake_labels = torch.full((B,), fake_label_value, device=device)
            gen_labels = torch.full((B,), gen_target_value, device=device)

            # discriminator training
            d_losses = []
            for _ in range(d_steps):
                noise_d = torch.normal(
                    0, 2 * torch.pi, (B, self.noise_dim), device=device
                )
                fake_data_d = self.G(noise_d).detach()

                self.D.zero_grad()
                outD_real = self.D(real_data).view(-1)  # logits
                outD_fake = self.D(fake_data_d).view(-1)  # logits, detached

                errD = criterion(outD_real, real_labels) + criterion(
                    outD_fake, fake_labels
                )
                errD.backward()
                optD.step()
                d_losses.append(errD.detach().item())

            D_loss = float(np.mean(d_losses))

            # generator training
            g_losses = []
            fake_data = None
            if optimizer_type == "spsa":
                # Update SPSA context to use current iteration's noise.
                noise_g = torch.normal(
                    0, 2 * torch.pi, (B, self.noise_dim), device=device
                )
                iter_gen_labels = torch.full((B,), gen_target_value, device=device)
                _spsa_ctx["noise"] = noise_g
                _spsa_ctx["gen_labels"] = iter_gen_labels

                new_params = optG.step(spsa_step_duration)
                # Sync the model weights to the updated SPSA params.
                self._set_G_params_flat(new_params)

                with torch.no_grad():
                    fake_data = self.G(noise_g)
                    outD_fake_for_G = self.D(fake_data).view(-1)
                    G_loss_val = criterion(outD_fake_for_G, iter_gen_labels).item()
                g_losses.append(G_loss_val)
            else:
                # freeze discriminator
                for p in self.D.parameters():
                    p.requires_grad_(False)

                for _ in range(g_steps):
                    noise_g = torch.normal(
                        0, 2 * torch.pi, (B, self.noise_dim), device=device
                    )
                    self.G.zero_grad()
                    fake_data = self.G(noise_g)
                    outD_fake_for_G = self.D(fake_data).view(-1)  # logits, NOT detached
                    G_loss = criterion(outD_fake_for_G, gen_labels)
                    G_loss.backward()
                    optG.step()
                    g_losses.append(G_loss.detach().item())

                # unfreeze
                for p in self.D.parameters():
                    p.requires_grad_(True)

            G_loss_val = float(np.mean(g_losses))

            # Similarity/diversity metrics based on skimage SSIM.
            with torch.no_grad():
                real_images = data.detach().cpu().numpy()
                if real_images.ndim == 4:
                    real_images = real_images[:, 0, :, :]
                elif real_images.ndim == 2:
                    real_images = real_images.reshape(
                        B, self.image_size, self.image_size
                    )

                fake_images = (
                    fake_data.detach()
                    .cpu()
                    .numpy()
                    .reshape(B, self.image_size, self.image_size)
                )
                real_images = np.clip(real_images, 0.0, 1.0)
                fake_images = np.clip(fake_images, 0.0, 1.0)
                similarity, diversity = get_metrics(real_images, fake_images)
                # Keep SSIM as the final column for downstream ranking scripts.
                ssim_prog.append((similarity, diversity, similarity))

            # log and display results
            D_loss_prog.append(D_loss)
            G_loss_prog.append(G_loss_val)

            if not silent:
                print("it", i)
                print("D_loss", D_loss)
                print("G_loss", G_loss_val)

            fake_samples = None
            step_interval = max(1, opt_iter_num // 100)
            should_capture = (i % step_interval == 0) or ((i + 1) % 100 == 0)
            if should_capture:
                with torch.no_grad():
                    fake_samples = self.G(fixed_noise.to(device)).detach().cpu()
                fake_progress.append(fake_samples)

            if callback is not None:
                callback(
                    i,
                    D_loss,
                    G_loss_val,
                    self.G.state_dict(),
                    self.D.state_dict(),
                    fake_samples,
                    optG,
                )

        final_G_params = self.G.state_dict()

        return D_loss_prog, G_loss_prog, final_G_params, fake_progress, ssim_prog
