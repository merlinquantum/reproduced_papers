"""Train + evaluate LatentQGAN (or a baseline) from a JSON config.

Public entry: ``train_and_evaluate(cfg, run_dir) -> dict``.

Recognised top-level config keys::

    model              "quantum" | "classical" | "random_decoder" | "merlin"
    digit              MNIST class to model (int in 0..9)
    ae_epochs          autoencoder training epochs
    ae_lr              autoencoder learning rate
    ae_batch_size      autoencoder batch size
    ae_data_size       max images for autoencoder training
    gan_iterations     QGAN iterations (per sample)
    gen_lr             generator learning rate
    disc_lr            discriminator learning rate
    gan_batch_size     batch size for GAN updates
    n_samples_eval     number of generated images for FD evaluation
    n_samples_real     number of real images for FD evaluation
    T, N, NA, L        quantum architecture params
    seed               random seed

Outputs to ``run_dir``::

    config_snapshot.json
    run.log
    metrics.json
    fd_curve.csv
    sample_real.png / sample_fake.png (small grids)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .autoencoder import Autoencoder
from .data import autoencoder_loader, gan_loader, load_mnist, subset_by_class
from .metrics import frechet_distance
from .qgan import ClassicalLatentGenerator, LatentDiscriminator, LatentQGenerator


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _log(run_dir: Path, msg: str) -> None:
    print(msg)
    with open(run_dir / "run.log", "a") as f:
        f.write(msg + "\n")


def _save_grid(imgs: torch.Tensor, path: Path, ncol: int = 8) -> None:
    """Save a tiny image grid as PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = imgs.shape[0]
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    if nrow == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(nrow * ncol):
        r, c = i // ncol, i % ncol
        axes[r, c].axis("off")
        if i < n:
            arr = imgs[i].squeeze().detach().cpu().numpy()
            axes[r, c].imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _build_generator(cfg: dict, T: int, N: int, NA: int, L: int):
    model = cfg.get("model", "quantum")
    if model == "quantum":
        return LatentQGenerator(T=T, N=N, NA=NA, L=L), "alpha"
    if model == "classical":
        hidden = cfg.get("classical_hidden_dim", 4)
        return ClassicalLatentGenerator(T=T, N=N, hidden_dim=hidden), "z"
    if model == "merlin":
        from .merlin_generator import MerlinLatentGenerator
        return MerlinLatentGenerator(T=T, N=N, NA=NA, L=L), "alpha"
    raise ValueError(f"unknown model: {model}")


def train_autoencoder(ae: Autoencoder, imgs: torch.Tensor, cfg: dict, run_dir: Path, device: torch.device) -> None:
    loader = autoencoder_loader(imgs, batch_size=cfg.get("ae_batch_size", 20), shuffle=True)
    opt = torch.optim.SGD(ae.parameters(), lr=cfg.get("ae_lr", 0.05))
    epochs = cfg.get("ae_epochs", 5)
    ae.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            recon = ae(x)
            loss = nn.functional.mse_loss(recon, x)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        _log(run_dir, f"[AE] epoch {ep+1}/{epochs} loss={total/n:.5f}")


def train_qgan(generator, discriminator: LatentDiscriminator, latents: torch.Tensor,
               cfg: dict, run_dir: Path, device: torch.device, decoder, real_imgs: np.ndarray,
               input_name: str = "alpha") -> list[dict]:
    """Train the QGAN (or classical-baseline GAN) on a single class.

    Returns the history of [{iter, loss_g, loss_d, fd?}] entries.
    """
    iters = cfg.get("gan_iterations", 200)
    gen_lr = cfg.get("gen_lr", 0.3)
    disc_lr = cfg.get("disc_lr", 0.01)
    bs = cfg.get("gan_batch_size", 1)
    eval_every = cfg.get("eval_every", 50)
    n_eval = cfg.get("n_samples_eval", 64)

    opt_g = torch.optim.SGD(generator.parameters(), lr=gen_lr)
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=disc_lr)
    bce = nn.BCELoss()

    history: list[dict] = []
    N_real = latents.shape[0]
    rng = np.random.default_rng(cfg.get("seed", 0))

    eps = 1e-7
    for it in range(iters):
        # Sample real latent rows
        idx = rng.integers(0, N_real, size=bs)
        x_real = latents[idx].to(device)
        # Sample fake latent rows
        if input_name == "alpha":
            noise = generator.sample_noise(bs, device=device)
            x_fake = generator(noise)
        else:
            noise = generator.sample_noise(bs, device=device)
            x_fake = generator(noise)
        # ---- Train discriminator ----
        opt_d.zero_grad()
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake.detach())
        loss_d = bce(d_real.clamp(eps, 1 - eps), torch.ones_like(d_real)) \
                 + bce(d_fake.clamp(eps, 1 - eps), torch.zeros_like(d_fake))
        loss_d.backward()
        opt_d.step()
        # ---- Train generator ----
        opt_g.zero_grad()
        d_fake = discriminator(x_fake)
        loss_g = bce(d_fake.clamp(eps, 1 - eps), torch.ones_like(d_fake))
        loss_g.backward()
        opt_g.step()
        entry = {"iter": it + 1, "loss_g": float(loss_g.item()), "loss_d": float(loss_d.item())}
        if (it + 1) % eval_every == 0 or (it + 1) == iters:
            entry["fd"] = _evaluate_fd(generator, decoder, real_imgs, n_eval, device, input_name)
            _log(run_dir, f"[GAN] it={it+1}/{iters} loss_g={entry['loss_g']:.4f} loss_d={entry['loss_d']:.4f} fd={entry['fd']:.3f}")
        history.append(entry)
    return history


def _evaluate_fd(generator, decoder, real_imgs: np.ndarray, n: int, device, input_name: str) -> float:
    generator.eval()
    decoder.eval()
    with torch.no_grad():
        noise = generator.sample_noise(n, device=device)
        latents = generator(noise)
        fake = decoder(latents).cpu().numpy()
    generator.train()
    decoder.train()
    return frechet_distance(real_imgs[:n], fake[:n])


def _random_decoder_baseline(decoder, latent_shape, n: int, device) -> np.ndarray:
    """RandomDecoder baseline from the paper: random normalised noise -> decoder."""
    with torch.no_grad():
        T, R = latent_shape
        z = torch.rand(n, T, R, device=device)
        z = z / z.sum(dim=-1, keepdim=True)
        imgs = decoder(z).cpu().numpy()
    return imgs


def train_and_evaluate(cfg: dict, run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2)
    _log(run_dir, f"Run dir: {run_dir}")
    _log(run_dir, f"Config: {json.dumps(cfg)}")

    seed = int(cfg.get("seed", 0))
    _seed(seed)
    device = torch.device(cfg.get("device", "cpu"))

    T = int(cfg.get("T", 5))
    N = int(cfg.get("N", 4))
    NA = int(cfg.get("NA", 1))
    L = int(cfg.get("L", 7))
    digit = int(cfg.get("digit", 0))

    # ---- Load data
    _log(run_dir, "Loading MNIST...")
    imgs, labels = load_mnist(cfg, train=True)
    ae_imgs = imgs
    if cfg.get("ae_data_size"):
        # Use a random subset across all classes.
        n = int(cfg["ae_data_size"])
        idx = torch.randperm(imgs.shape[0])[:n]
        ae_imgs = imgs[idx]
    digit_imgs = subset_by_class(imgs, labels, digit, cfg.get("gan_data_size"))
    _log(run_dir, f"AE data: {ae_imgs.shape}, digit {digit} data: {digit_imgs.shape}")

    # ---- Train autoencoder (with optional cache by (seed, ae_data_size, ae_epochs))
    _log(run_dir, "Training autoencoder...")
    ae = Autoencoder(T=T, NG=N - NA).to(device)
    cache_dir = Path(cfg.get("ae_cache_dir", Path(__file__).resolve().parent.parent / "outdir" / "_ae_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"ae_seed{seed}_size{cfg.get('ae_data_size', 'all')}_ep{cfg.get('ae_epochs', 1)}_bs{cfg.get('ae_batch_size', 20)}_lr{cfg.get('ae_lr', 0.05)}.pt"
    cache_path = cache_dir / cache_key
    t0 = time.time()
    if cache_path.exists() and not cfg.get("force_retrain_ae", False):
        _log(run_dir, f"Loading cached AE from {cache_path}")
        ae.load_state_dict(torch.load(cache_path, map_location=device))
    else:
        train_autoencoder(ae, ae_imgs, cfg, run_dir, device)
        torch.save(ae.state_dict(), cache_path)
        _log(run_dir, f"Cached AE to {cache_path}")
    t_ae = time.time() - t0
    _log(run_dir, f"AE training: {t_ae:.1f}s")

    # ---- Encode digit images to latent space
    ae.eval()
    with torch.no_grad():
        latents = ae.encoder(digit_imgs.to(device)).cpu()
    _log(run_dir, f"latent shape: {tuple(latents.shape)}")

    # ---- Build generator + discriminator
    if cfg.get("model") == "random_decoder":
        generator = None
        input_name = "z"
        discriminator = None
        n_gen_params = 0
        n_disc_params = 0
        _log(run_dir, "Skipping generator/discriminator (random_decoder baseline).")
    else:
        generator, input_name = _build_generator(cfg, T, N, NA, L)
        generator = generator.to(device)
        discriminator = LatentDiscriminator(latent_dim=T * (2 ** (N - NA)),
                                            h1=cfg.get("disc_h1", 64),
                                            h2=cfg.get("disc_h2", 16)).to(device)
        n_gen_params = sum(p.numel() for p in generator.parameters())
        n_disc_params = sum(p.numel() for p in discriminator.parameters())
        _log(run_dir, f"Generator params: {n_gen_params}, Discriminator params: {n_disc_params}")

    # ---- Real-image reference set for FD (with no decoder pass to avoid double penalty)
    n_real = cfg.get("n_samples_real", min(256, digit_imgs.shape[0]))
    real_for_fd = digit_imgs[:n_real].cpu().numpy()

    # ---- Train QGAN
    _log(run_dir, "Training QGAN...")
    t1 = time.time()
    if cfg.get("model") == "random_decoder":
        history = []
        # No GAN training; just evaluate the random-decoder baseline.
        fake = _random_decoder_baseline(ae.decoder, (T, 2 ** (N - NA)),
                                        cfg.get("n_samples_eval", 64), device)
        fd_final = frechet_distance(real_for_fd[: cfg.get("n_samples_eval", 64)], fake)
        _log(run_dir, f"[RandomDecoder] fd={fd_final:.3f}")
        history.append({"iter": 0, "fd": fd_final})
    else:
        history = train_qgan(generator, discriminator, latents, cfg, run_dir, device,
                             ae.decoder, real_for_fd, input_name=input_name)
    t_gan = time.time() - t1
    _log(run_dir, f"QGAN training: {t_gan:.1f}s")

    # ---- Save fd curve csv
    if history:
        with open(run_dir / "fd_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "loss_g", "loss_d", "fd"])
            for e in history:
                writer.writerow([e.get("iter", ""), e.get("loss_g", ""), e.get("loss_d", ""), e.get("fd", "")])

    # ---- Generate samples for inspection
    if cfg.get("model") == "random_decoder":
        fake_imgs_t = torch.tensor(_random_decoder_baseline(ae.decoder, (T, 2 ** (N - NA)), 16, device))
    else:
        with torch.no_grad():
            noise = generator.sample_noise(16, device=device)
            latents_gen = generator(noise)
            fake_imgs_t = ae.decoder(latents_gen).cpu()
    _save_grid(fake_imgs_t, run_dir / "sample_fake.png")
    _save_grid(digit_imgs[:16].cpu(), run_dir / "sample_real.png")

    # ---- Final FD and assemble metrics
    final_fd = next((e["fd"] for e in reversed(history) if "fd" in e), None)
    best_fd = min((e["fd"] for e in history if "fd" in e), default=None)
    metrics = {
        "model": cfg.get("model"),
        "digit": digit,
        "ae_train_time_s": t_ae,
        "gan_train_time_s": t_gan,
        "gen_params": n_gen_params,
        "disc_params": n_disc_params,
        "final_fd": final_fd,
        "best_fd": best_fd,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return {"test_metrics": metrics, "history": history, "run_dir": str(run_dir)}
