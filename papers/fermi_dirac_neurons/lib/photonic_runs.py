"""Runner tasks of the Phase 4 photonic adaptation (MerLin) and its Kerr ablation.

* ``photonic_train`` (P4-L): photonic counterparts of Fig. 7 / C6 / C8. Every
  model of ``cfg["models"]`` is trained on the same seeded data with
  :func:`lib.training.train_models` (same full-batch gradient descent,
  ``history.csv`` and log events as the qubit runs). After training, the
  MerLin neuron's validation predictions are re-evaluated under threshold and
  postselected (UNBUNCHED) detection and with finite PNR shots.
* ``photonic_gradient_variance`` (E13): variance over random parameters of
  d f and d L for the MerLin neuron and the commuting photon-count neuron.
* ``kerr_gradient_variance`` (E13-K): the same variance for a passive
  Hamiltonian plus a cross-Kerr chain, by dense Fock-space diagonalization.
  **Not MerLin**: an interacting H has no interferometer eigenbasis.

Config blocks (written by ``utils/generate_configs.py``)::

    "photonic": {"n_modes", "n_photons", "target_range",
                 "train_states": {"set", "n_states"},
                 "validation_states": {"set", "n_states"},
                 "evaluation": {"readouts": [...], "shots": [...]}}
    "photonic_gradient_variance": {"n_modes", "n_photons", "draws",
                 "energy_ranges": [...], "temperatures": [...],
                 "inputs": [...], "target_range"}
    "kerr_gradient_variance": {"n_modes", "n_photons", "draws", "init_range",
                 "kerr": [...], "temperature", "inputs": [...],
                 "parameters": [...]}
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from . import photonic as ph
from .activations import divided_differences, make_activation
from .data import (
    STREAM_INITIALIZATION,
    STREAM_SHOTS,
    STREAM_TARGET,
    STREAM_TRAIN_STATES,
    STREAM_VALIDATION_STATES,
    random_stream,
    selection_split,
)
from .trainability import bootstrap_variance_ci, hermitian_eigh
from .training import train_models

logger = logging.getLogger(__name__)

STREAM_PHOTONIC_GRADIENT = 500
STREAM_KERR_GRADIENT = 600
PROGRESS_EVERY = 20


def _describe_photonic_dataset(
    block: dict, d: int, teacher: ph.PassiveTeacher, splits: dict
) -> dict:
    return {
        "name": (
            f"synthetic photonic regression: N={block['n_photons']} photons in "
            f"m={block['n_modes']} modes (Fock dimension {d}); target "
            "Tr[tanh(H*/T) rho] for a nearest-neighbour passive H* with "
            f"zeta ~ U[-{block['target_range']}, {block['target_range']}]"
        ),
        "splits": {
            name: {
                # The Phase 5 selection split is carved out of the training states.
                "state_set": block[
                    f"{'train' if name == 'selection' else name}_states"
                ]["set"],
                "n_states": int(len(data.targets)),
            }
            for name, data in splits.items()
        },
        "preprocessing": "none (amplitudes on the MerLin FOCK basis)",
        "subset": "full",
        "teacher_labels": teacher.labels,
    }


def run_photonic_training(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """P4-L: train the photonic neuron and its baselines on one seeded dataset."""
    block = cfg["photonic"]
    n_modes, n_photons = int(block["n_modes"]), int(block["n_photons"])
    temperature = float(cfg["physics"]["temperature"])
    d = len(ph.fock_basis(n_modes, n_photons))
    seed = cfg["seed"]
    teacher = ph.build_teacher(
        n_modes,
        n_photons,
        float(block["target_range"]),
        random_stream(seed, STREAM_TARGET),
    )
    splits = {}
    for split, stream in (
        ("train", STREAM_TRAIN_STATES),
        ("validation", STREAM_VALIDATION_STATES),
    ):
        state_cfg = block[f"{split}_states"]
        states = ph.build_states(
            state_cfg["set"],
            n_modes,
            n_photons,
            state_cfg["n_states"],
            random_stream(seed, stream),
        )
        splits[split] = ph.PhotonicDataset(states, teacher.targets(states, temperature))
    if "selection_fraction" in block:
        # Phase 5: learning rates are selected on states held out of training.
        train_index, selection_index = selection_split(
            len(splits["train"].targets), float(block["selection_fraction"]), seed
        )
        full = splits["train"]
        splits = {
            "train": ph.PhotonicDataset(
                full.states[train_index], full.targets[train_index]
            ),
            "selection": ph.PhotonicDataset(
                full.states[selection_index], full.targets[selection_index]
            ),
            "validation": splits["validation"],
        }
    dataset_description = _describe_photonic_dataset(block, d, teacher, splits)
    logger.info(
        "DATASET_READY | m=%d | N=%d | d=%d | train=%s n=%d | validation=%s n=%d | target variance train=%.4g",
        n_modes,
        n_photons,
        d,
        block["train_states"]["set"],
        len(splits["train"].targets),
        block["validation_states"]["set"],
        len(splits["validation"].targets),
        float(np.var(splits["train"].targets)),
    )
    training_cfg = cfg["training"]
    models, initial = {}, {}
    for index, model_cfg in enumerate(cfg["models"]):
        name = model_cfg["name"]
        if name in models:
            raise ValueError(f"Duplicate model name {name!r}")
        model = ph.build_photonic_model(model_cfg, n_modes, n_photons, temperature)
        rng = random_stream(seed, STREAM_INITIALIZATION + index)
        weights = rng.uniform(
            float(training_cfg["init_low"]),
            float(training_cfg["init_high"]),
            size=model.n_parameters,
        )
        if isinstance(model, ph.MerlinPassiveNeuron):
            phase_low, phase_high = training_cfg["phase_init"]
            weights[: model.n_phases] = rng.uniform(
                float(phase_low), float(phase_high), size=model.n_phases
            )
        models[name], initial[name] = model, weights
        logger.info(
            "MODEL_READY | name=%s | kind=%s | parameters=%d | activation=%s",
            name,
            model_cfg["kind"],
            model.n_parameters,
            model_cfg.get("activation", "tanh"),
        )
    result = train_models(
        models,
        initial,
        splits["train"],
        splits,
        training_cfg,
        "regression",
        run_dir / "history.csv",
        learning_rates={
            model["name"]: float(model["learning_rate"])
            for model in cfg["models"]
            if "learning_rate" in model
        },
    )
    ablations = _detector_and_shot_ablations(
        cfg, models, result["final_parameters"], splits["validation"]
    )
    metrics = {
        "task": "photonic_train",
        "experiment": cfg["experiment"]["id"],
        "task_kind": "regression",
        "seed": seed,
        "n_modes": n_modes,
        "n_photons": n_photons,
        "fock_dimension": d,
        "temperature": temperature,
        "target": {"labels": teacher.labels, "zeta": teacher.zeta.tolist()},
        "models": {
            name: {
                "n_parameters": model.n_parameters,
                "parameter_labels": model.labels,
                "initial_parameters": initial[name].tolist(),
                "final_parameters": result["final_parameters"][name],
                "last_logged_train_loss": result["final_train_loss_before_last_update"][
                    name
                ],
                "final": result["final_metrics"][name],
                "learning_rate": result["learning_rates"][name],
                "diverged_at_epoch": result["diverged_at_epoch"][name],
                "training_seconds": result["training_seconds"][name],
            }
            for name, model in models.items()
        },
        "optimizer": result["optimizer"],
        "readout_ablations": ablations,
        "evaluation_history": result["evaluations"],
        "artifacts": {"history": "history.csv"},
    }
    return metrics, dataset_description


def _detector_and_shot_ablations(
    cfg: dict, models: dict, final_parameters: dict, validation: ph.PhotonicDataset
) -> dict:
    """Validation MSE of every MerLin model under each detector model and shot count."""
    evaluation = cfg["photonic"]["evaluation"]
    records = {}
    for index, (name, model) in enumerate(models.items()):
        if not isinstance(model, ph.MerlinPassiveNeuron):
            continue
        theta = np.array(final_parameters[name])
        record = {}
        for readout in evaluation["readouts"]:
            logger.info(
                "EVALUATION_STARTED | split=validation | model=%s | readout=%s",
                name,
                readout,
            )
            predictions = model.readout_outputs(theta, validation, readout)
            record[f"mse_{readout}"] = float(
                np.mean((predictions - validation.targets) ** 2)
            )
            logger.info(
                "EVALUATION_COMPLETED | split=validation | model=%s | readout=%s | mse=%.10g",
                name,
                readout,
                record[f"mse_{readout}"],
            )
        rng = random_stream(cfg["seed"], STREAM_SHOTS + index)
        for shots in evaluation["shots"]:
            predictions = model.sampled_outputs(theta, validation, int(shots), rng)
            record[f"mse_shots{int(shots)}"] = float(
                np.mean((predictions - validation.targets) ** 2)
            )
            logger.info(
                "EVALUATION_COMPLETED | split=validation | model=%s | readout=pnr | shots=%d | mse=%.10g",
                name,
                shots,
                record[f"mse_shots{int(shots)}"],
            )
        records[name] = record
    return records


def _input_states(
    kinds: list[str], n_modes: int, n_photons: int, rng, fixed: dict
) -> np.ndarray:
    d = len(ph.fock_basis(n_modes, n_photons))
    rows = []
    for kind in kinds:
        if kind == "fock_first":
            rows.append(fixed["fock_first"])
        elif kind == "haar_interferometer":
            rows.append(ph.haar_interferometer_states(n_modes, n_photons, 1, rng)[0])
        elif kind == "haar_full":
            rows.append(ph.haar_full_states(d, 1, rng)[0])
        else:
            raise ValueError(f"Unknown input kind {kind!r}; use {ph.INPUT_KINDS}")
    return np.array(rows)


def _fixed_inputs(n_modes: int, n_photons: int) -> dict:
    basis = ph.fock_basis(n_modes, n_photons)
    state = np.zeros(len(basis), dtype=complex)
    state[basis.index(ph.first_fock_state(n_modes, n_photons))] = 1.0
    return {"fock_first": state}


def _summarize(samples: dict, keys: tuple[str, ...], rng) -> list[dict]:
    records = []
    for key, values in samples.items():
        values = np.array(values)
        low, high = bootstrap_variance_ci(values, rng)
        records.append(
            {
                **dict(zip(keys, key)),
                "draws": len(values),
                "mean": float(values.mean()),
                "variance": float(values.var(ddof=1)),
                "variance_ci95": [low, high],
                "mean_square": float(np.mean(values**2)),
            }
        )
    return records


def run_photonic_gradient_variance(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """E13: Var over random parameters of d f and d L for the MerLin and photon-count neurons."""
    import torch

    block = cfg["photonic_gradient_variance"]
    n_modes, n_photons = int(block["n_modes"]), int(block["n_photons"])
    d = len(ph.fock_basis(n_modes, n_photons))
    draws = int(block["draws"])
    inputs = list(block["inputs"])
    temperatures = [float(value) for value in block["temperatures"]]
    energy_ranges = [float(value) for value in block["energy_ranges"]]
    rng = np.random.default_rng([int(cfg["seed"]), STREAM_PHOTONIC_GRADIENT])
    teacher = ph.build_teacher(n_modes, n_photons, float(block["target_range"]), rng)
    merlin_model = ph.MerlinPassiveNeuron(n_modes, n_photons, "tanh", 1.0)
    count_model = ph.PhotonCountNeuron(n_modes, n_photons, 1.0)
    occupation = merlin_model.occupation.numpy()
    phase_positions = {"phase_first": 0, "phase_last": merlin_model.n_phases - 1}
    fixed = _fixed_inputs(n_modes, n_photons)
    logger.info(
        "DATASET_READY | random parameters (phases ~ U[0, 2pi), energies ~ U[-a, a], a in %s) and inputs %s (no dataset) | m=%d | N=%d | d=%d | draws=%d",
        energy_ranges,
        ",".join(inputs),
        n_modes,
        n_photons,
        d,
        draws,
    )
    samples: dict = {}

    def add(key, value):
        samples.setdefault(key, []).append(float(value))

    for draw in range(1, draws + 1):
        phases = rng.uniform(0.0, 2.0 * math.pi, size=merlin_model.n_phases)
        unit_energies = rng.uniform(-1.0, 1.0, size=n_modes)
        unit_count_weights = rng.uniform(-1.0, 1.0, size=count_model.n_parameters)
        states = _input_states(inputs, n_modes, n_photons, rng, fixed)
        data = ph.PhotonicDataset(states, np.zeros(len(states)))
        merlin_model.set_phases(phases)
        for parameter in merlin_model.phase_parameters:
            parameter.grad = None
            parameter.requires_grad_(True)
        probabilities = merlin_model.probabilities(data)
        probabilities_np = probabilities.detach().numpy()
        count_probabilities = data.count_probabilities()
        for temperature in temperatures:
            targets = teacher.targets(states, temperature)
            for energy_range in energy_ranges:
                energies = occupation @ (energy_range * unit_energies)
                values = np.tanh(energies / temperature)
                slopes = (1.0 - values**2) / temperature
                count_activation = make_activation("tanh", temperature)
                count_pre = count_model.design @ (energy_range * unit_count_weights)
                count_values = count_activation.value(count_pre)
                count_slopes = count_activation.derivative(count_pre)
                for row, kind in enumerate(inputs):
                    output = probabilities_np[row] @ values
                    residual = output - targets[row]
                    gradients = {
                        "eps0": probabilities_np[row] @ (slopes * occupation[:, 0])
                    }
                    weighted = probabilities[row] @ torch.as_tensor(values)
                    phase_gradient = torch.autograd.grad(
                        weighted, merlin_model.phase_parameters, retain_graph=True
                    )
                    flat = torch.cat([g.reshape(-1) for g in phase_gradient]).numpy()
                    for name, position in phase_positions.items():
                        gradients[name] = flat[position]
                    for name, value in gradients.items():
                        cell = ("photonic_tanh", kind, temperature, energy_range, name)
                        add((*cell, "output"), value)
                        add((*cell, "loss"), 2.0 * residual * value)
                    count_output = count_probabilities[row] @ count_values
                    count_gradient = count_probabilities[row] @ (
                        count_slopes * count_model.design[:, 0]
                    )
                    cell = ("photon_count_tanh", kind, temperature, energy_range, "w0")
                    add((*cell, "output"), count_gradient)
                    add(
                        (*cell, "loss"),
                        2.0 * (count_output - targets[row]) * count_gradient,
                    )
        if draw % PROGRESS_EVERY == 0 or draw == draws:
            logger.info("PROGRESS | draw=%d/%d", draw, draws)
    records = _summarize(
        samples,
        ("model", "input", "temperature", "energy_range", "parameter", "quantity"),
        rng,
    )
    for record in records:
        logger.info(
            "EVALUATION_COMPLETED | model=%s | input=%s | T=%g | a=%g | parameter=%s | quantity=%s | variance=%.4e",
            record["model"],
            record["input"],
            record["temperature"],
            record["energy_range"],
            record["parameter"],
            record["quantity"],
            record["variance"],
        )
    metrics = {
        "task": "photonic_gradient_variance",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        "n_modes": n_modes,
        "n_photons": n_photons,
        "fock_dimension": d,
        "n_phases": merlin_model.n_phases,
        "teacher": {"labels": teacher.labels, "zeta": teacher.zeta.tolist()},
        "records": records,
    }
    dataset = {
        "name": "none (random parameters and random input states)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": "n/a",
    }
    return metrics, dataset


def _sparse_fock_operator(single_particle: np.ndarray, n_photons: int):
    """Sparse N-photon matrix of sum_ij h_ij a_i^dag a_j."""
    import scipy.sparse

    n_modes = single_particle.shape[0]
    operators = ph.hopping_operators(n_modes, n_photons)
    d = len(ph.fock_basis(n_modes, n_photons))
    total = scipy.sparse.csr_matrix((d, d), dtype=complex)
    for i, j in zip(*np.nonzero(single_particle)):
        total = total + single_particle[i, j] * operators[i][j]
    return total


def run_kerr_gradient_variance(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """E13-K (not MerLin): Var of d f for passive H + kappa * sum_i n_i n_{i+1}."""
    block = cfg["kerr_gradient_variance"]
    n_modes, n_photons = int(block["n_modes"]), int(block["n_photons"])
    d = len(ph.fock_basis(n_modes, n_photons))
    draws = int(block["draws"])
    inputs = list(block["inputs"])
    kerr_strengths = [float(value) for value in block["kerr"]]
    temperature = float(block["temperature"])
    activation = make_activation("tanh", temperature)
    labels, generators = ph.passive_generators(n_modes)
    unknown = set(block["parameters"]) - set(labels)
    if unknown:
        raise ValueError(f"Unknown generator labels {sorted(unknown)}; have {labels}")
    # Generators stay sparse (J of them); only the measured ones are densified.
    sparse_generators = [_sparse_fock_operator(g, n_photons) for g in generators]
    measured = {
        name: sparse_generators[labels.index(name)].toarray()
        for name in block["parameters"]
    }
    kerr_diagonal = ph.nearest_neighbour_kerr(n_modes, n_photons)
    rng = np.random.default_rng([int(cfg["seed"]), STREAM_KERR_GRADIENT])
    fixed = _fixed_inputs(n_modes, n_photons)
    logger.info(
        "DATASET_READY | dense Fock-space simulation (not MerLin) | theta ~ U[-%g, %g] over %d passive generators | kappa in %s | inputs %s | m=%d | N=%d | d=%d | draws=%d",
        block["init_range"],
        block["init_range"],
        len(labels),
        kerr_strengths,
        ",".join(inputs),
        n_modes,
        n_photons,
        d,
        draws,
    )
    samples: dict = {}
    for draw in range(1, draws + 1):
        theta = rng.uniform(-block["init_range"], block["init_range"], size=len(labels))
        states = _input_states(inputs, n_modes, n_photons, rng, fixed)
        passive = sum(
            value * generator for value, generator in zip(theta, sparse_generators)
        ).toarray()
        for kappa in kerr_strengths:
            eigenvalues, eigenvectors = hermitian_eigh(
                passive + np.diag(kappa * kerr_diagonal)
            )
            divided = divided_differences(activation, eigenvalues)
            rotated_states = states @ eigenvectors.conj()
            for name, generator in measured.items():
                rotated = eigenvectors.conj().T @ generator @ eigenvectors
                kernel = divided * rotated
                for row, kind in enumerate(inputs):
                    vector = rotated_states[row]
                    value = float(np.real(vector.conj() @ kernel @ vector))
                    samples.setdefault((kappa, kind, name, "output"), []).append(value)
        if draw % PROGRESS_EVERY == 0 or draw == draws:
            logger.info("PROGRESS | draw=%d/%d", draw, draws)
    records = _summarize(samples, ("kerr", "input", "parameter", "quantity"), rng)
    for record in records:
        logger.info(
            "EVALUATION_COMPLETED | kappa=%g | input=%s | parameter=%s | variance=%.4e",
            record["kerr"],
            record["input"],
            record["parameter"],
            record["variance"],
        )
    metrics = {
        "task": "kerr_gradient_variance",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        "simulator": "dense numpy Fock-space diagonalization (not MerLin)",
        "n_modes": n_modes,
        "n_photons": n_photons,
        "fock_dimension": d,
        "temperature": temperature,
        "records": records,
    }
    dataset = {
        "name": "none (random parameters and random input states)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": "n/a",
    }
    return metrics, dataset
