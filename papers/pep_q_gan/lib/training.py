"""Conditional WGAN-GP, matching the author's losses and optimizer settings.

Local RNG streams and a checkpointable batch stream replace the Hydra/DDP shell.
Training preserves all rows of the supplied headerless file.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .artifacts import replace_file, write_json
from .gan import build_gan, load_training_dataset
from .legacy_priors import GaussianPrior

_INITIALIZATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrainingConfig:
    prior: str = "gaussian"
    seed: int = 46
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    discriminator_iterations: int = 5
    gradient_penalty: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 100000
    device: str = "cpu"
    rng_mode: str = "private_cpu"

    def __post_init__(self):
        if self.rng_mode not in ("private_cpu", "global_device"):
            raise ValueError("Unknown RNG mode")
        if self.prior not in ("gaussian", "dist", "indist"):
            raise ValueError("Unknown prior")
        if min(self.batch_size, self.discriminator_iterations, self.max_steps) < 1:
            raise ValueError(
                "Batch size, critic iterations and max steps must be positive"
            )
        if (
            self.warmup_steps < 0
            or self.learning_rate <= 0
            or self.gradient_penalty < 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer or penalty setting")


class BatchStream:
    """Shuffle without replacement, drop_last, and save the exact next batch."""

    def __init__(self, dataset, batch_size, seed):
        if not 1 <= batch_size <= len(dataset):
            raise ValueError(
                "batch_size must be positive and no larger than the dataset"
            )
        self.peptides = dataset.peptide_tokenizer.batch_encode(dataset.peptides, 9)
        table = {
            hla: dataset.hla_tokenizer.encode(hla, 34) for hla in set(dataset.hla_types)
        }
        self.hlas = torch.stack([table[hla] for hla in dataset.hla_types])
        self.batch_size = batch_size
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(self.peptides), generator=self.rng)
        self.cursor = 0
        self.epoch = 0

    def next(self, device):
        if self.cursor + self.batch_size > len(self.order):
            self.order = torch.randperm(len(self.peptides), generator=self.rng)
            self.cursor = 0
            self.epoch += 1
        ids = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return self.peptides[ids].to(device), self.hlas[ids].to(device)

    def state_dict(self):
        return {
            "order": self.order,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "rng": self.rng.get_state(),
        }

    def load_state_dict(self, state):
        self.order = state["order"].cpu()
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        self.rng.set_state(state["rng"].cpu())


def gradient_penalty(critic, real, fake, hla, epsilon):
    interp = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    scores = critic(interp, hla)
    gradients = torch.autograd.grad(
        scores, interp, torch.ones_like(scores), create_graph=True
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def make_optimizer(model, cfg):
    groups = [
        {
            "params": [p for p in model.parameters() if p.dim() != 1],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.dim() == 1],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.5, 0.999))


def checked_grad_norm(model):
    norms = [
        p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None
    ]
    norm = torch.stack(norms).sum().sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite model gradient")
    return float(norm)


class ClassicalTrainer:
    """Shared classical WGAN optimizer; received photonic pools can supply its prior."""

    def __init__(self, config: TrainingConfig, dataset, prior=None):
        if config.prior in ("dist", "indist") and (
            prior is None or prior.condition != config.prior
        ):
            raise ValueError(
                "Photonic priors require a matching explicitly supplied pool"
            )
        self.config = config
        if config.rng_mode == "global_device":
            if threading.current_thread() is not threading.main_thread():
                raise ValueError(
                    "Global device RNG requires an isolated process, not stream threads"
                )
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        self.generator, self.critic = build_gan(config.seed, config.device)
        self.g_opt = make_optimizer(self.generator, config)
        self.d_opt = make_optimizer(self.critic, config)
        prior_cls = GaussianPrior
        self.prior = (
            prior
            if prior is not None
            else prior_cls(seed=config.seed, rng_mode=config.rng_mode)
        )
        self.epsilon_rng = torch.Generator().manual_seed(config.seed + 1)
        self.data = BatchStream(dataset, config.batch_size, config.seed)
        self.step = 0

    def train_step(self):
        cfg = self.config
        self.generator.train()
        self.critic.train()
        lr = (
            min(self.step / cfg.warmup_steps, 1.0) * cfg.learning_rate
            if cfg.warmup_steps
            else cfg.learning_rate
        )
        for opt in (self.g_opt, self.d_opt):
            for group in opt.param_groups:
                group["lr"] = lr
        for p in self.critic.parameters():
            p.requires_grad_(True)
        d_metrics = []
        for _ in range(cfg.discriminator_iterations):
            ids, hla = self.data.next(cfg.device)
            with torch.no_grad():
                fake = self.generator(
                    self.prior.sample(len(ids), device=cfg.device), hla
                ).softmax(-1)
                real = F.one_hot(ids, 22).to(fake.dtype)
            self.d_opt.zero_grad(set_to_none=True)
            real_scores = self.critic(real, hla)
            fake_scores = self.critic(fake, hla)
            wass = fake_scores.mean() - real_scores.mean()
            epsilon = (
                torch.rand(len(ids), 1, 1, device=cfg.device)
                if cfg.rng_mode == "global_device"
                else torch.rand(len(ids), 1, 1, generator=self.epsilon_rng).to(
                    cfg.device
                )
            )
            gp = gradient_penalty(self.critic, real, fake, hla, epsilon)
            loss = wass + cfg.gradient_penalty * gp
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite critic loss")
            loss.backward()
            norm = checked_grad_norm(self.critic)
            self.d_opt.step()
            d_metrics.append(
                [float(loss.detach()), float(wass.detach()), float(gp.detach()), norm]
            )
        for p in self.critic.parameters():
            p.requires_grad_(False)
        ids, hla = self.data.next(cfg.device)
        self.g_opt.zero_grad(set_to_none=True)
        fake = self.generator(
            self.prior.sample(len(ids), device=cfg.device), hla
        ).softmax(-1)
        g_loss = -self.critic(fake, hla).mean()
        if not torch.isfinite(g_loss):
            raise FloatingPointError("Nonfinite generator loss")
        g_loss.backward()
        g_norm = checked_grad_norm(self.generator)
        self.g_opt.step()
        self.step += 1
        means = [sum(row[i] for row in d_metrics) / len(d_metrics) for i in range(4)]
        return {
            "step": self.step,
            "epoch": self.data.epoch,
            "lr": lr,
            "g_loss": float(g_loss.detach()),
            "d_loss": means[0],
            "wasserstein_loss": means[1],
            "gradient_penalty": means[2],
            "d_grad_norm": means[3],
            "g_grad_norm": g_norm,
            "token_entropy": float(
                -(fake.detach() * fake.detach().clamp_min(1e-12).log()).sum(-1).mean()
            ),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "global_step": self.step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.critic.state_dict(),
            "G_optimizer_state_dict": self.g_opt.state_dict(),
            "D_optimizer_state_dict": self.d_opt.state_dict(),
            "batch_stream": self.data.state_dict(),
            "epsilon_rng": self.epsilon_rng.get_state(),
        }
        if hasattr(self.prior, "state_dict"):
            payload["prior_state"] = self.prior.state_dict()
        else:
            payload["prior_rng"] = self.prior._rng.get_state()
        if self.config.rng_mode == "global_device":
            payload["torch_rng_state"] = torch.get_rng_state()
            if str(self.config.device).startswith("cuda"):
                payload["cuda_rng_state"] = torch.cuda.get_rng_state(self.config.device)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        replace_file(temporary, path)

    def restore(self, path):
        state = torch.load(path, map_location=self.config.device, weights_only=True)
        previous = {k: v for k, v in state["config"].items() if k != "device"}
        previous.setdefault("rng_mode", "private_cpu")
        current = {k: v for k, v in asdict(self.config).items() if k != "device"}
        if previous != current:
            raise ValueError("Resume config differs from checkpoint")
        self.step = state["global_step"]
        self.generator.load_state_dict(state["generator_state_dict"])
        self.critic.load_state_dict(state["discriminator_state_dict"])
        self.g_opt.load_state_dict(state["G_optimizer_state_dict"])
        self.d_opt.load_state_dict(state["D_optimizer_state_dict"])
        self.data.load_state_dict(state["batch_stream"])
        if "prior_state" in state:
            self.prior.load_state_dict(state["prior_state"])
        else:
            self.prior._rng.set_state(state["prior_rng"].cpu())
        self.epsilon_rng.set_state(state["epsilon_rng"].cpu())
        if self.config.rng_mode == "global_device":
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if str(self.config.device).startswith("cuda"):
                torch.cuda.set_rng_state(
                    state["cuda_rng_state"].cpu(), self.config.device
                )


def run_training(
    config,
    data_dir,
    run_dir,
    *,
    resume=False,
    stop_after=None,
    save_every=100,
    status_every=1,
    dataset=None,
    print_progress=True,
    prior=None,
):
    if min(save_every, status_every) < 1 or (stop_after is not None and stop_after < 1):
        raise ValueError("save_every and stop_after must be positive")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError("Use a fresh run directory or --resume")
    if dataset is None:
        dataset = load_training_dataset(data_dir)
    fingerprint = hashlib.sha256(
        (Path(data_dir) / "HLA_pos9mers.txt").read_bytes()
    ).hexdigest()
    data_files = {
        name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
        for name in ("HLA_pos9mers.txt", "MHC_pseudo_HLA_only.dat")
    }
    metadata = {
        "config": asdict(config),
        "training_rows": len(dataset),
        "data_sha256": fingerprint,
        "data_files_sha256": data_files,
        "torch_version": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "differences_from_author": [
            "Keep all 105971 headerless rows (upstream skips first row)",
            (
                "Global device RNG shared by prior and GP, as upstream; batch RNG remains private"
                if config.rng_mode == "global_device"
                else "Independent RNG streams; not seed-identical to upstream"
            ),
            "Checkpointable pretokenized shuffle instead of worker DataLoader",
        ],
        "evaluation": "NetMHCpan evaluation not performed",
    }
    if prior is not None:
        metadata["latent_prior"] = prior.metadata
    if resume:
        old = json.loads((run_dir / "config.json").read_text())
        if old["data_files_sha256"] != data_files:
            raise ValueError("Training data changed since checkpoint")
    else:
        write_json(run_dir / "config.json", metadata)
    # Model initialization temporarily uses the global CPU RNG; serialize it
    # while each trainer's subsequent noise, batches and epsilon use private RNGs.
    with _INITIALIZATION_LOCK:
        trainer = ClassicalTrainer(config, dataset, prior=prior)
    if resume:
        trainer.restore(latest)
    else:
        trainer.save(latest)
    initial = trainer.step
    start = time.monotonic()
    target = (
        min(config.max_steps, initial + stop_after)
        if stop_after is not None
        else config.max_steps
    )
    write_json(
        run_dir / "status.json",
        {
            "state": "running",
            "step": trainer.step,
            "target_steps": config.max_steps,
            "pid": os.getpid(),
            "device": config.device,
        },
    )
    # Resume may leave metric rows newer than the last checkpoint after a crash.
    # Preserve them as evidence; new rows include the starting checkpoint step.
    try:
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
            while trainer.step < target:
                tick = time.monotonic()
                row = trainer.train_step()
                row.update(seconds=time.monotonic() - tick, resumed_from=initial)
                log.write(json.dumps(row) + "\n")
                status = {
                    "state": "running",
                    "step": trainer.step,
                    "target_steps": config.max_steps,
                    "pid": os.getpid(),
                    "device": config.device,
                    "elapsed_this_run_seconds": time.monotonic() - start,
                    "last_metrics": row,
                }
                if trainer.step == initial + 1 or trainer.step % status_every == 0:
                    log.flush()
                    write_json(run_dir / "status.json", status)
                if trainer.step % save_every == 0:
                    log.flush()
                    trainer.save(latest)
                if trainer.step % 2000 == 0:
                    trainer.save(run_dir / f"checkpoint_step_{trainer.step}.pt")
                if print_progress and (
                    trainer.step == initial + 1 or trainer.step % 10 == 0
                ):
                    print(json.dumps(row), flush=True)
        trainer.save(latest)
        status = {
            "state": "complete" if trainer.step == config.max_steps else "paused",
            "step": trainer.step,
            "target_steps": config.max_steps,
            "elapsed_this_run_seconds": time.monotonic() - start,
        }
        write_json(run_dir / "status.json", status)
        return status
    except BaseException as error:
        # A partially applied step cannot be resumed exactly: keep last completed checkpoint.
        write_json(
            run_dir / "status.json",
            {
                "state": "interrupted",
                "step": trainer.step,
                "error": str(error),
                "resume_from_last_completed_checkpoint": True,
            },
        )
        raise
