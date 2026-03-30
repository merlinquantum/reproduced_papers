from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

# Set up paths based on the runner location
RUNNER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNNER_DIR.parent  # photonic_QGAN/
REPO_ROOT = PROJECT_ROOT.parent.parent  # reproduced_papers/


def _resolve_digit_list(cfg: dict) -> list[int]:
    digits_cfg = cfg.get("digits", {})
    explicit = digits_cfg.get("digits")
    if explicit:
        return [int(d) for d in explicit]
    start = int(digits_cfg.get("digit_start", 1))
    end = int(digits_cfg.get("digit_end", 9))
    return list(range(start, end + 1))


def _coerce_str_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _coerce_int_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [int(item) for item in loaded]
    return [int(loaded)]


def _load_config_grid(path_value) -> list[dict]:
    if not path_value:
        return []
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "config_grid" in payload:
        payload = payload["config_grid"]
    if not isinstance(payload, list):
        raise ValueError("ideal.config_grid_path must point to a JSON list")
    return payload


def _arch_is_valid_for_modes(arch: list[str], mode_count: int) -> bool:
    for token in arch:
        if token.startswith("enc[") and token.endswith("]"):
            idx_text = token[4:-1]
            try:
                idx = int(idx_text)
            except ValueError:
                return False
            if idx < 0 or idx >= mode_count:
                return False
    return True


def _setup_arch_grid() -> list[dict]:
    return [
        {
            "setup": "setup_a",
            "noise_dim": 1,
            "arch": ["var", "enc[2]", "var"],
        },
        {
            "setup": "setup_b",
            "noise_dim": 3,
            "arch": ["var", "enc[0]", "var", "enc[2]", "var", "enc[4]", "var"],
        },
        {
            "setup": "setup_c",
            "noise_dim": 1,
            "arch": ["var", "var", "enc[2]", "var", "var"],
        },
        {
            "setup": "setup_d",
            "noise_dim": 2,
            "arch": ["var", "enc[1]", "var", "enc[3]", "var"],
        },
    ]


def _setup_arch_from_label(setup_label: str) -> dict | None:
    key = str(setup_label).strip().lower()
    for item in _setup_arch_grid():
        if item["setup"] == key:
            return {"noise_dim": int(item["noise_dim"]), "arch": list(item["arch"])}
    return None


def _coerce_input_state(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    if (
        "," not in text
        and "[" not in text
        and "]" not in text
        and all(ch in "01" for ch in text)
    ):
        return [int(ch) for ch in text]
    return _coerce_int_list(value)


def _default_ideal_grid() -> list[dict]:
    # User-selected setup family:
    # A: var, enc[2], var
    # B: var, enc[0], var, enc[2], var, enc[4], var
    # C: var, var, enc[2], var, var
    # D: var, enc[1], var, enc[3], var
    setup_arch_grid = [
        {"noise_dim": item["noise_dim"], "arch": item["arch"]}
        for item in _setup_arch_grid()
    ]
    input_grid_4modes = [
        {"input_state": [1, 1, 1, 1], "gen_count": 2, "pnr": True},
        {"input_state": [1, 1, 1, 1], "gen_count": 4, "pnr": False},
        {"input_state": [1, 0, 1, 1], "gen_count": 4, "pnr": True},
    ]
    input_grid_5modes = [
        {"input_state": [0, 1, 0, 1, 0], "gen_count": 4, "pnr": False},
        {"input_state": [1, 0, 1, 0, 1], "gen_count": 2, "pnr": True},
    ]
    input_grid_8modes = [
        {"input_state": [0, 0, 1, 0, 0, 1, 0, 0], "gen_count": 2, "pnr": False},
    ]

    config_grid: list[dict] = []
    for inp in input_grid_4modes + input_grid_5modes + input_grid_8modes:
        mode_count = len(inp["input_state"])
        for arch in setup_arch_grid:
            if not _arch_is_valid_for_modes(arch["arch"], mode_count):
                continue
            config = inp.copy()
            config.update(arch)
            config_grid.append(config)
    return config_grid


def _ideal_setup_label(arch: list[str]) -> str:
    arch_tuple = tuple(arch)
    for item in _setup_arch_grid():
        if tuple(item["arch"]) == arch_tuple:
            return item["setup"]
    return "setup_other"


def _input_state_tag(input_state: list[int]) -> str:
    return "".join(str(int(x)) for x in input_state)


def _resolve_csv_path(
    csv_path: str | Path, data_root: str | Path | None, project_dir: Path
) -> Path:
    """Resolve CSV file path with multiple fallback strategies.

    Priority order:
    1. Absolute paths are returned as-is
    2. repo_root/data/<csv_path> (default data directory)
    3. data_root/<csv_path> if data_root is provided
    4. project_dir/<csv_path> (legacy fallback)
    """
    path = Path(csv_path)
    if path.is_absolute():
        return path

    candidate_paths: list[Path] = []

    # Priority 1: Default data directory at repo root
    candidate_paths.append((REPO_ROOT / "data" / path).resolve())

    # Priority 2: Explicit data_root if provided
    if data_root:
        candidate_paths.append((Path(data_root) / path).resolve())

    # Priority 3: Relative to project_dir (legacy fallback)
    candidate_paths.append((project_dir / path).resolve())

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "CSV file not found; tried: "
        + ", ".join(str(candidate) for candidate in candidate_paths)
    )


def _prepare_dataset(
    csv_path: str | Path,
    label: int | None,
    batch_size: int,
    opt_iter_num: int,
    data_root: str | Path | None,
    project_dir: Path,
    log,
):
    import torch
    from torch.utils.data import RandomSampler
    from torchvision import transforms

    from papers.shared.photonic_QGAN.digits import DigitsDataset

    resolved_csv = _resolve_csv_path(csv_path, data_root, project_dir)
    dataset = DigitsDataset(
        csv_file=str(resolved_csv),
        label=label if label is not None else 0,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=batch_size * opt_iter_num
    )
    log.debug(
        "Prepared dataset csv_path={} label={} size={} batch_size={} opt_iter_num={}",
        resolved_csv,
        label,
        len(dataset),
        batch_size,
        opt_iter_num,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )


def _run_qgan(
    cfg: dict,
    run_dir: Path,
    run_cfg: dict,
    dataloader,
    write_to_disk: bool,
    lrD: float,
    lrG: float,
    opt_iter_num: int,
    train_params: dict,
    log_every: int,
    show_progress: bool,
    log,
) -> None:
    import numpy as np
    import perceval as pcvl
    from lib.qgan import QGAN

    model_cfg = cfg.get("model", {})
    image_size = int(model_cfg.get("image_size", 8))
    batch_size = int(model_cfg.get("batch_size", 4))
    lossy = bool(model_cfg.get("lossy", False))
    remote_token = model_cfg.get("remote_token")
    use_clements = bool(model_cfg.get("use_clements", False))
    sim = bool(model_cfg.get("sim", False))
    generator_type = str(model_cfg.get("generator_type", "photonic")).strip().lower()

    gen_arch = run_cfg["arch"]
    noise_dim = int(run_cfg["noise_dim"])
    input_state = run_cfg["input_state"]
    pnr = bool(run_cfg["pnr"])
    gen_count = int(run_cfg["gen_count"])

    log.info(
        "Starting QGAN run image_size={} gen_count={} noise_dim={} batch_size={} pnr={} lossy={} use_clements={} sim={} generator_type={}",
        image_size,
        gen_count,
        noise_dim,
        batch_size,
        pnr,
        lossy,
        use_clements,
        sim,
        generator_type,
    )
    log.info(
        "Generator config patches={} modes={} input_state={} arch={}",
        gen_count,
        len(input_state),
        input_state,
        gen_arch,
    )
    iterator = (
        tqdm(dataloader, desc="iter", leave=False) if show_progress else dataloader
    )
    progress_images_dir = run_dir / "generated_every_100"
    if write_to_disk:
        progress_images_dir.mkdir(parents=True, exist_ok=True)

    def _log_progress(
        i,
        d_loss,
        g_loss,
        _g_params,
        _d_state,
        fake_samples,
        _optG,
    ):
        if log_every > 0 and (i + 1) % log_every == 0:
            log.info("Iteration {} D_loss={} G_loss={}", i + 1, d_loss, g_loss)
        if write_to_disk and fake_samples is not None and (i + 1) % 100 == 0:
            try:
                import matplotlib.pyplot as plt

                if hasattr(fake_samples, "detach"):
                    sample_arr = fake_samples.detach().cpu().numpy()
                else:
                    sample_arr = np.asarray(fake_samples)
                if sample_arr.ndim > 1:
                    sample = sample_arr[0].reshape(image_size, image_size)
                else:
                    sample = sample_arr.reshape(image_size, image_size)
                plt.imshow(sample, cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                out_path = progress_images_dir / f"fake_iter_{i + 1:04d}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
            except Exception as exc:
                log.warning("Failed to save iteration image at {}: {}", i + 1, exc)

    start = time.perf_counter()
    qgan = QGAN(
        image_size,
        gen_count,
        gen_arch,
        pcvl.BasicState(input_state),
        noise_dim,
        batch_size,
        pnr,
        lossy,
        remote_token=remote_token,
        use_clements=use_clements,
        sim=sim,
        generator_type=generator_type,
    )
    (
        D_loss_progress,
        G_loss_progress,
        G_params_progress,
        fake_data_progress,
        ssim_progress,
    ) = qgan.fit(
        iterator,
        lrD,
        lrG,
        opt_iter_num,
        train_params=train_params,
        silent=True,
        callback=_log_progress if log_every > 0 else None,
    )

    if write_to_disk:
        fake_rows = []
        for snapshot in fake_data_progress:
            if hasattr(snapshot, "detach"):
                arr = snapshot.detach().cpu().numpy()
            else:
                arr = np.asarray(snapshot)
            fake_rows.append(arr.reshape(-1))
        fake_progress_2d = np.stack(fake_rows, axis=0)
        np.savetxt(
            run_dir / "fake_progress.csv",
            fake_progress_2d,
            delimiter=",",
        )
        np.savetxt(
            run_dir / "loss_progress.csv",
            np.column_stack((D_loss_progress, G_loss_progress)),
            delimiter=",",
            header="D_loss, G_loss",
        )
        if ssim_progress:
            ssim_array = np.asarray(ssim_progress, dtype=float)
            iterations = np.arange(1, ssim_array.shape[0] + 1, dtype=int)
            np.savetxt(
                run_dir / "ssim_progress.csv",
                np.column_stack((iterations, ssim_array)),
                delimiter=",",
                header="iter,similarity,diversity,ssim",
            )
        try:
            import torch

            torch.save(G_params_progress, run_dir / "G_params_progress.pt")
            flat_params = []
            for tensor in G_params_progress.values():
                if hasattr(tensor, "detach"):
                    flat_params.append(tensor.detach().cpu().numpy().reshape(-1))
                else:
                    flat_params.append(np.asarray(tensor).reshape(-1))
            if flat_params:
                np.savetxt(
                    run_dir / "G_params_progress.csv",
                    np.concatenate(flat_params).reshape(1, -1),
                    delimiter=",",
                )
        except Exception as exc:
            log.warning("Failed to save generator state dict: {}", exc)
        try:
            import matplotlib.pyplot as plt

            last_snapshot = fake_data_progress[-1]
            if hasattr(last_snapshot, "detach"):
                last_snapshot = last_snapshot.detach().cpu().numpy()
            else:
                last_snapshot = np.asarray(last_snapshot)
            if last_snapshot.ndim > 1:
                sample = last_snapshot[0].reshape(image_size, image_size)
            else:
                sample = last_snapshot.reshape(image_size, image_size)
            plt.imshow(sample, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(run_dir / "fake_progress_last.png", dpi=150)
            plt.close()
            log.debug(
                "Saved image preview under {}", run_dir / "fake_progress_last.png"
            )
        except Exception as exc:
            log.warning("Failed to save image preview: {}", exc)
        log.debug("Saved outputs under %s", run_dir)
    duration = time.perf_counter() - start
    log.info("Completed run at {} duration={:.2f}s", run_dir, duration)


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_and_evaluate(cfg, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(run_dir / "qgan.log", level="DEBUG")
    logger.debug(
        "Resolved config: {}",
        json.dumps(cfg, indent=2, default=str),
    )

    mode = cfg.get("run", {}).get("mode", "smoke")
    if mode == "smoke":
        artifact = run_dir / "done.txt"
        artifact.write_text(
            "Smoke run complete. Use --mode digits or --mode ideal for training.\n",
            encoding="utf-8",
        )
        logger.info("Wrote placeholder artifact: %s", artifact)
        return

    if mode not in {"digits", "ideal", "hp_study"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    run_cfg = cfg.get("run", {})
    write_to_disk = bool(run_cfg.get("write_to_disk", True))
    runs = int(run_cfg.get("runs", 1))
    show_progress = bool(run_cfg.get("progress", False))
    log_every = int(run_cfg.get("log_every", 0))
    logger.info(
        "Run configuration mode={} runs={} write_to_disk={} progress={} log_every={}",
        mode,
        runs,
        write_to_disk,
        show_progress,
        log_every,
    )

    if mode == "hp_study":
        from lib.hp_study import run_halving_grid_study

        run_halving_grid_study(cfg, run_dir, logger)
        return

    project_dir = run_dir.parent.parent.resolve()
    dataset_cfg = cfg.get("dataset", {})
    csv_path = dataset_cfg.get("csv_path", "photonic_QGAN/optdigits_csv.csv")

    # Resolve data_root: use config value, DATA_DIR env var, or default to repo_root/data
    import os

    data_root_cfg = cfg.get("data_root")
    if data_root_cfg:
        data_root = Path(data_root_cfg).expanduser()
    elif "DATA_DIR" in os.environ:
        data_root = Path(os.environ["DATA_DIR"]).expanduser()
    else:
        data_root = REPO_ROOT / "data"

    logger.info("Dataset csv_path={}", csv_path)

    training_cfg = cfg.get("training", {})
    if mode == "digits":
        train_cfg = training_cfg.get("digits", {})
    else:
        train_cfg = training_cfg.get("ideal", {})
    opt_iter_num = int(train_cfg.get("opt_iter_num", 0))
    lrD = float(train_cfg.get("lrD", 0.002))
    lrG = float(train_cfg.get("lrG", lrD))
    optimizer = str(train_cfg.get("optimizer", "adam")).strip().lower()
    d_optimizer = str(train_cfg.get("d_optimizer", "adam")).strip().lower()
    adam_beta1 = float(train_cfg.get("adam_beta1", 0.5))
    adam_beta2 = float(train_cfg.get("adam_beta2", 0.999))
    real_label = float(train_cfg.get("real_label", 0.9))
    fake_label = float(train_cfg.get("fake_label", 0.0))
    gen_target = float(train_cfg.get("gen_target", real_label))
    d_steps = int(train_cfg.get("d_steps", 1))
    g_steps = int(train_cfg.get("g_steps", 1))
    if opt_iter_num <= 0:
        raise ValueError("Training iterations must be positive for digits/ideal modes.")
    if d_steps <= 0 or g_steps <= 0:
        raise ValueError("d_steps and g_steps must be positive for digits/ideal modes.")

    train_params = {
        "optimizer": optimizer,
        "d_optimizer": d_optimizer,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "real_label": real_label,
        "fake_label": fake_label,
        "gen_target": gen_target,
        "d_steps": d_steps,
        "g_steps": g_steps,
    }

    if optimizer == "spsa":
        spsa_iter_num = int(train_cfg.get("spsa_iter_num", 10500))
        train_params["spsa_iter_num"] = spsa_iter_num
        if "spsa_a" in train_cfg:
            train_params["spsa_a"] = float(train_cfg["spsa_a"])
        if "spsa_k" in train_cfg:
            train_params["spsa_k"] = int(train_cfg["spsa_k"])

    logger.info(
        "Training params opt_iter_num={} lrD={} lrG={} optimizer={} d_optimizer={} betas=({}, {}) labels=(real={}, fake={}, gen={}) steps=(d={}, g={})",
        opt_iter_num,
        lrD,
        lrG,
        optimizer,
        d_optimizer,
        adam_beta1,
        adam_beta2,
        real_label,
        fake_label,
        gen_target,
        d_steps,
        g_steps,
    )

    if mode == "digits":
        digits_cfg = cfg.get("digits", {})
        digits = _resolve_digit_list(cfg)
        logger.info("Digits mode digits={}", digits)
        base_config = {
            "noise_dim": int(digits_cfg.get("noise_dim", 1)),
            "arch": _coerce_str_list(digits_cfg.get("arch")),
            "input_state": _coerce_int_list(digits_cfg.get("input_state")),
            "gen_count": int(digits_cfg.get("gen_count", 1)),
            "pnr": bool(digits_cfg.get("pnr", False)),
        }
        logger.debug("Digits base config: {}", json.dumps(base_config, indent=2))
        model_cfg = cfg.get("model", {})
        batch_size = int(model_cfg.get("batch_size", 4))
        logger.info("Model batch_size={}", batch_size)
        output_root = run_dir / "digits"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for digit in digits:
            dataloader = _prepare_dataset(
                csv_path,
                digit,
                batch_size,
                opt_iter_num,
                data_root,
                project_dir,
                logger,
            )
            config_path = output_root / f"config_{digit}"
            config_path.mkdir(parents=True, exist_ok=True)
            config_payload = base_config.copy()
            config_payload.update({"digit": digit})
            _write_config(config_path / "config.json", config_payload)
            logger.info("Digit {} output_dir={}", digit, config_path)

            run_num = 0
            attempt = 0
            while run_num < runs and attempt < 1000:
                attempt += 1
                run_num += 1
                save_path = config_path / f"run_{run_num}"
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info("Digit {} run {}/{}", digit, run_num, runs)
                try:
                    _run_qgan(
                        cfg,
                        save_path,
                        base_config,
                        dataloader,
                        write_to_disk,
                        lrD,
                        lrG,
                        opt_iter_num,
                        train_params,
                        log_every,
                        show_progress,
                        logger,
                    )
                except Exception as exc:
                    logger.exception("Run failed for digit {}: {}", digit, exc)
                    shutil.rmtree(save_path, ignore_errors=True)
                    run_num -= 1

        return

    ideal_cfg = cfg.get("ideal", {})
    # Resolve digit list: ideal.digits (list) takes priority over ideal.digit (single int)
    _ideal_digits_raw = ideal_cfg.get("digits")
    if _ideal_digits_raw:
        ideal_digits = [int(d) for d in _ideal_digits_raw]
    else:
        ideal_digits = [int(ideal_cfg.get("digit", 0))]

    if ideal_cfg.get("config_grid_path"):
        config_grid = _load_config_grid(ideal_cfg.get("config_grid_path"))
        logger.info(
            "Ideal mode config_grid_path={} size={}",
            ideal_cfg.get("config_grid_path"),
            len(config_grid),
        )
    elif ideal_cfg.get("config_grid"):
        config_grid = list(ideal_cfg["config_grid"])
        logger.info("Ideal mode config_grid_size={}", len(config_grid))
    elif ideal_cfg.get("setup") and ideal_cfg.get("input_state") is not None:
        setup_label = str(ideal_cfg.get("setup")).strip().lower()
        setup_cfg = _setup_arch_from_label(setup_label)
        if setup_cfg is None:
            raise ValueError(
                f"Unknown ideal.setup={ideal_cfg.get('setup')}. Expected one of: setup_a, setup_b, setup_c, setup_d."
            )
        input_state = _coerce_input_state(ideal_cfg.get("input_state"))
        if not input_state:
            raise ValueError(
                "ideal.input_state must be a non-empty list, csv string, or bitstring like '01010'."
            )
        mode_count = len(input_state)
        if not _arch_is_valid_for_modes(setup_cfg["arch"], mode_count):
            raise ValueError(
                f"Selected setup {setup_label} is incompatible with input_state of length {mode_count}."
            )
        gen_count = int(ideal_cfg.get("gen_count", 4 if mode_count == 5 else 2))
        pnr = bool(ideal_cfg.get("pnr", False))
        config_grid = [
            {
                "input_state": input_state,
                "gen_count": gen_count,
                "pnr": pnr,
                "noise_dim": setup_cfg["noise_dim"],
                "arch": setup_cfg["arch"],
            }
        ]
        logger.info(
            "Ideal mode selected setup={} input_state={} gen_count={} pnr={}",
            setup_label,
            input_state,
            gen_count,
            pnr,
        )
    else:
        config_grid = _default_ideal_grid()
        logger.info("Ideal mode default config_grid_size={}", len(config_grid))

    model_cfg = cfg.get("model", {})
    batch_size = int(model_cfg.get("batch_size", 4))
    logger.info("Model batch_size={} ideal_digits={}", batch_size, ideal_digits)

    for ideal_digit in ideal_digits:
        logger.info("--- Ideal mode digit={} ---", ideal_digit)
        dataloader = _prepare_dataset(
            csv_path,
            ideal_digit,
            batch_size,
            opt_iter_num,
            data_root,
            project_dir,
            logger,
        )

        output_root = run_dir / f"ideal-{ideal_digit}"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for config_num, config in enumerate(config_grid):
            setup_dir = output_root / _ideal_setup_label(config["arch"])
            state_tag = _input_state_tag(config["input_state"])
            config_path = setup_dir / f"config_{config_num}_input_{state_tag}"
            config_path.mkdir(parents=True, exist_ok=True)
            _write_config(config_path / "config.json", config)
            logger.debug(
                "Ideal config {}: {}", config_num, json.dumps(config, indent=2)
            )

            run_num = 0
            attempt = 0
            while run_num < runs and attempt < 1000:
                attempt += 1
                run_num += 1
                save_path = config_path / f"run_{run_num}"
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Ideal digit={} config {} run {}/{}",
                    ideal_digit,
                    config_num,
                    run_num,
                    runs,
                )
                try:
                    _run_qgan(
                        cfg,
                        save_path,
                        config,
                        dataloader,
                        write_to_disk,
                        lrD,
                        lrG,
                        opt_iter_num,
                        train_params,
                        log_every,
                        show_progress,
                        logger,
                    )
                except Exception as exc:
                    logger.exception(
                        "Run failed for digit={} config {}: {}",
                        ideal_digit,
                        config_num,
                        exc,
                    )
                    shutil.rmtree(save_path, ignore_errors=True)
                    run_num -= 1
