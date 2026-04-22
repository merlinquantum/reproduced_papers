from __future__ import annotations


VALID_MODEL_TYPES = {"A", "B", "C", "D", "E", "F", "VisionTransformer", "OrthoFNN"}
VALID_PROFILES = {"full", "lite"}
VALID_CIRCUIT_FAMILIES = {"generic", "butterfly"}
VALID_PRECISION_MODES = {"baseline", "gpu_friendly"}
VALID_TRAIN_SUBSET_MODES = {"random", "stratified"}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _positive_int(cfg: dict, key: str, default: int) -> int:
    value = int(cfg.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}.")
    return value


def validate_run_config(cfg: dict) -> dict:
    """Validate a resolved experiment config and return a normalized copy."""
    normalized = dict(cfg)

    model_type = normalized.get("model_type", "B")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected one of {sorted(VALID_MODEL_TYPES)}."
        )

    profile = normalized.get("profile", "full")
    if profile not in VALID_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of {sorted(VALID_PROFILES)}."
        )
    normalized["profile"] = profile

    circuit_family = normalized.get("circuit_family", "generic")
    if circuit_family not in VALID_CIRCUIT_FAMILIES:
        raise ValueError(
            f"Unknown circuit_family '{circuit_family}'. Expected one of {sorted(VALID_CIRCUIT_FAMILIES)}."
        )
    normalized["circuit_family"] = circuit_family

    precision_mode = normalized.get("precision_mode", "baseline")
    if precision_mode not in VALID_PRECISION_MODES:
        raise ValueError(
            f"Unknown precision_mode '{precision_mode}'. Expected one of {sorted(VALID_PRECISION_MODES)}."
        )
    normalized["precision_mode"] = precision_mode

    image_embed_grayscale = bool(normalized.get("image_embed_grayscale", False))
    if image_embed_grayscale and model_type != "OrthoFNN":
        raise ValueError("image_embed_grayscale is only valid for model_type='OrthoFNN'.")
    normalized["image_embed_grayscale"] = image_embed_grayscale

    train_subset_size = normalized.get("train_subset_size")
    if train_subset_size in ("", None):
        train_subset_size = None
    elif int(train_subset_size) <= 0:
        raise ValueError(f"train_subset_size must be positive, got {train_subset_size}.")
    else:
        train_subset_size = int(train_subset_size)
    normalized["train_subset_size"] = train_subset_size

    train_subset_mode = normalized.get("train_subset_mode", "stratified")
    if train_subset_mode not in VALID_TRAIN_SUBSET_MODES:
        raise ValueError(
            f"Unknown train_subset_mode '{train_subset_mode}'. "
            f"Expected one of {sorted(VALID_TRAIN_SUBSET_MODES)}."
        )
    normalized["train_subset_mode"] = train_subset_mode

    if "train_subset_seed" in normalized and normalized["train_subset_seed"] is not None:
        normalized["train_subset_seed"] = int(normalized["train_subset_seed"])

    data_regime = normalized.get("data_regime")
    if data_regime in (None, ""):
        if train_subset_size is None:
            data_regime = "standard"
        elif train_subset_size == 1080:
            data_regime = "retina_sized_train"
        else:
            data_regime = f"train_subset_{train_subset_size}"
    normalized["data_regime"] = str(data_regime)

    compound_readout = normalized.get("compound_readout", "cross_only")
    if compound_readout == "full_sector" and model_type != "D":
        raise ValueError("compound_readout='full_sector' is only valid for model_type='D'.")

    img_size = _positive_int(normalized, "img_size", 28)
    patch_size = _positive_int(normalized, "patch_size", 7)
    embed_dim = _positive_int(normalized, "embed_dim", 16)
    _positive_int(normalized, "n_layers", 4)
    _positive_int(normalized, "epochs", 100)
    _positive_int(normalized, "batch_size", 32)

    if img_size % patch_size != 0:
        raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}.")

    if circuit_family == "butterfly":
        if model_type in {"A", "B", "C", "OrthoFNN"} and not _is_power_of_two(embed_dim):
            raise ValueError(
                f"Butterfly circuit_family requires embed_dim to be a power of two for model {model_type}; got {embed_dim}."
            )

        if model_type in {"D", "E"}:
            n_patches = (img_size // patch_size) ** 2
            use_cls_token = bool(normalized.get("use_cls_token", True))
            total_modes = n_patches + int(use_cls_token) + embed_dim
            if not _is_power_of_two(total_modes):
                raise ValueError(
                    f"Butterfly circuit_family requires total_modes to be a power of two for model {model_type}; "
                    f"got total_modes={total_modes} from n_patches={n_patches}, use_cls_token={use_cls_token}, embed_dim={embed_dim}."
                )

        if model_type == "F":
            n_regions = _positive_int(normalized, "n_regions_per_side", 2) ** 2
            n_patches_per_region = _positive_int(normalized, "n_patches_per_side", 2) ** 2
            total_modes = n_regions + n_patches_per_region + embed_dim
            if not _is_power_of_two(total_modes):
                raise ValueError(
                    f"Butterfly circuit_family requires total_modes to be a power of two for model F; "
                    f"got total_modes={total_modes}."
                )

    return normalized
