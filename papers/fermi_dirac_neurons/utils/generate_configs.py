"""Write the named experiment configs into ``configs/``.

Run from the paper folder::

    python3 utils/generate_configs.py

Each config overlays ``configs/defaults.json``. Suffixes:

* ``_original``: the protocol of the authors' released notebooks (commit
  9c2094c), which produced Figs. 7, 8 and Table II. For Fig. 9 (no code
  released) the settings are inferred from Fig. 7 (LOG.md D9).
* ``_papertext``: the protocol as written in the paper text (LOG.md D1, D3,
  D6): paper state sets, softplus target, activation-sign labels.
* ``_classical``: fair baselines B2 (classical neuron on the same Pauli
  features) and B3 (quantum linear model), on exactly the ``_original`` data
  (same seed and data block).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# The E15 term sets are sampled with the paper's own library.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Paper-text p values for sqrt(p)|0..0> + sqrt(1-p)|1..1> are unspecified
# ("various choices of p in (0, 1)", p. 21); this grid is our assumption.
PAPER_P_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]
HAAR_VALIDATION = {"set": "haar", "n_states": 500}


def _model(name, kind, term_set, include_identity, activation):
    return {
        "name": name,
        "kind": kind,
        "term_set": term_set,
        "include_identity": include_identity,
        "activation": activation,
    }


def _squared_loss_data(
    n_qubits, train_set, target_term_set, include_identity, function
):
    train_states = {"set": train_set}
    if train_set == "paper_squared_loss":
        train_states["p_values"] = PAPER_P_VALUES
    return {
        "n_qubits": n_qubits,
        "train_states": train_states,
        "validation_states": HAAR_VALIDATION,
        "label_rule": "regression",
        "target": {
            "term_set": target_term_set,
            "include_identity": include_identity,
            "low": -2.0,
            "high": 2.0,
            "function": function,
        },
    }


def _training(loss, learning_rate, epochs, eval_every):
    return {
        "loss": loss,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "eval_every": eval_every,
        "init_low": -0.5,
        "init_high": 0.5,
    }


def figure7_configs() -> dict:
    """Fig. 7 (p. 20): squared loss, TFIM vs IM, tanh and softplus, 2 and 7 qubits."""
    configs = {}
    for n_qubits in (2, 7):
        for activation in ("tanh", "softplus"):
            stem = f"fig7_{activation}_{n_qubits}q"
            panel = {"tanh": {2: "a", 7: "b"}, "softplus": {2: "c", 7: "d"}}[
                activation
            ][n_qubits]
            # Notebook protocol: softplus_nqubits.ipynb trains T*softplus(H/T^2)
            # against Tr[exp(H*/T) rho] (D1, D2); mse_nqubits.ipynb is exact tanh.
            model_activation = "softplus_code" if activation == "softplus" else "tanh"
            target_function = "exp" if activation == "softplus" else "tanh"
            original = {
                "description": f"Fig. 7{panel}: squared loss, {activation}, TFIM (quantum) vs IM (classical), {n_qubits} qubits, notebook protocol",
                "experiment": {"task": "train", "id": stem},
                "data": _squared_loss_data(
                    n_qubits, "code_squared_loss", "tfim", True, target_function
                ),
                "models": [
                    _model(
                        "quantum_tfim", "hamiltonian", "tfim", True, model_activation
                    ),
                    _model(
                        "classical_im", "hamiltonian", "ising", True, model_activation
                    ),
                ],
                "training": _training("squared", 0.1, 2000, 100),
            }
            configs[f"{stem}_original"] = original
            papertext = copy.deepcopy(original)
            papertext["description"] = (
                f"Fig. 7{panel}: squared loss, {activation}, TFIM vs IM, {n_qubits} qubits, paper-text protocol (states p. 21, target Eq. (120))"
            )
            papertext["experiment"]["id"] = f"{stem}_papertext"
            papertext["data"] = _squared_loss_data(
                n_qubits, "paper_squared_loss", "tfim", True, activation
            )
            for model in papertext["models"]:
                model["activation"] = activation
            configs[f"{stem}_papertext"] = papertext
            classical = copy.deepcopy(original)
            classical["description"] = (
                f"Fig. 7{panel} fair baselines on the notebook data: B2 classical neuron on TFIM Pauli features ({model_activation}), B3 quantum linear model, {n_qubits} qubits"
            )
            classical["experiment"]["id"] = f"{stem}_classical"
            classical["models"] = [
                _model(
                    "b2_pauli_features",
                    "pauli_features",
                    "tfim",
                    True,
                    model_activation,
                ),
                _model("b3_quantum_linear", "pauli_features", "tfim", True, "linear"),
            ]
            configs[f"{stem}_classical"] = classical
    return configs


def figure8_configs() -> dict:
    """Fig. 8 (p. 20) and Table II (p. 21): logistic loss, Heisenberg vs FCIM, 2-7 qubits."""
    configs = {}
    for n_qubits in range(2, 8):
        stem = f"fig8_logistic_{n_qubits}q"
        original = {
            "description": f"Fig. 8 / Table II, {n_qubits} qubits: logistic loss, Heisenberg vs FCIM, notebook protocol (1000 Haar train, labels sign Tr[H* rho])",
            "experiment": {"task": "train", "id": stem},
            "data": {
                "n_qubits": n_qubits,
                "train_states": {"set": "haar", "n_states": 1000},
                "validation_states": HAAR_VALIDATION,
                "label_rule": "sign_linear",
                "target": {
                    "term_set": "heisenberg",
                    "include_identity": False,
                    "low": -2.0,
                    "high": 2.0,
                    "function": "linear",
                },
            },
            "models": [
                _model(
                    "quantum_heisenberg", "hamiltonian", "heisenberg", False, "tanh"
                ),
                _model("classical_fcim", "hamiltonian", "fcim", False, "tanh"),
            ],
            # The notebook's gradient is (1/T) times the true gradient (D5), so
            # its eta = 0.1 equals eta = 0.05 with the exact gradient.
            "training": _training("logistic", 0.05, 500, 20),
        }
        configs[f"{stem}_original"] = original
        papertext = copy.deepcopy(original)
        papertext["description"] = (
            f"Fig. 8 / Table II, {n_qubits} qubits: logistic loss, Heisenberg vs FCIM, paper-text protocol (X/Y/Z product-basis train states, labels sign Tr[g_T(H*) rho], eta = 0.1)"
        )
        papertext["experiment"]["id"] = f"{stem}_papertext"
        papertext["data"]["train_states"] = {"set": "pauli_basis"}
        papertext["data"]["label_rule"] = "sign_activation"
        papertext["data"]["target"]["function"] = "tanh"
        papertext["training"]["learning_rate"] = 0.1
        configs[f"{stem}_papertext"] = papertext
        classical = copy.deepcopy(original)
        classical["description"] = (
            f"Fig. 8 / Table II fair baseline on the notebook data, {n_qubits} qubits: B2 classical logistic regression on the Heisenberg Pauli features"
        )
        classical["experiment"]["id"] = f"{stem}_classical"
        classical["models"] = [
            _model("b2_pauli_features", "pauli_features", "heisenberg", False, "tanh")
        ]
        configs[f"{stem}_classical"] = classical
    return configs


def figure9_configs() -> dict:
    """Fig. 9 (p. 23): classical vs quantum linear vs quantum tanh, 2 and 3 qubits.

    No code was released; settings follow Fig. 7 (state set, eta, init, T)
    with 10 000 epochs. Panels (c), (d) use the Heisenberg target and model
    without identity and FCIM as classical model (plot labels, LOG.md D9).
    """
    configs = {}
    for model_name, n_qubits, panel in (
        ("tfim", 2, "a"),
        ("tfim", 3, "b"),
        ("heis", 2, "c"),
        ("heis", 3, "d"),
    ):
        stem = f"fig9_{model_name}_{n_qubits}q"
        if model_name == "tfim":
            term_set, include_identity, classical_set = "tfim", True, "ising"
        else:
            term_set, include_identity, classical_set = "heisenberg", False, "fcim"
        original = {
            "description": f"Fig. 9{panel}: squared loss, tanh target, {model_name} tanh vs linear vs {classical_set}, {n_qubits} qubits (settings inferred, no code released)",
            "experiment": {"task": "train", "id": stem},
            "data": _squared_loss_data(
                n_qubits, "code_squared_loss", term_set, include_identity, "tanh"
            ),
            "models": [
                _model(
                    f"quantum_{model_name}_tanh",
                    "hamiltonian",
                    term_set,
                    include_identity,
                    "tanh",
                ),
                _model(
                    f"quantum_{model_name}_linear",
                    "hamiltonian",
                    term_set,
                    include_identity,
                    "linear",
                ),
                _model(
                    f"classical_{classical_set}",
                    "hamiltonian",
                    classical_set,
                    include_identity,
                    "tanh",
                ),
            ],
            "training": _training("squared", 0.1, 10000, 1000),
        }
        configs[f"{stem}_original"] = original
        classical = copy.deepcopy(original)
        classical["description"] = (
            f"Fig. 9{panel} fair baseline B2: classical tanh neuron on the {term_set} Pauli features, {n_qubits} qubits"
        )
        classical["experiment"]["id"] = f"{stem}_classical"
        classical["models"] = [
            _model(
                "b2_pauli_features",
                "pauli_features",
                term_set,
                include_identity,
                "tanh",
            )
        ]
        configs[f"{stem}_classical"] = classical
        # Pre-declared sensitivity variants from the Fig. 9 gap diagnosis
        # (LOG.md): the settings that move the tanh convergence rate most.
        faster_learning = copy.deepcopy(original)
        faster_learning["description"] = (
            f"Fig. 9{panel} sensitivity: eta = 0.3 instead of 0.1"
        )
        faster_learning["experiment"]["id"] = f"{stem}_sens_eta03"
        faster_learning["training"]["learning_rate"] = 0.3
        configs[f"{stem}_sens_eta03"] = faster_learning
        smaller_target = copy.deepcopy(original)
        smaller_target["description"] = (
            f"Fig. 9{panel} sensitivity: target coefficients zeta ~ U[-1, 1] instead of U[-2, 2]"
        )
        smaller_target["experiment"]["id"] = f"{stem}_sens_zeta1"
        smaller_target["data"]["target"]["low"] = -1.0
        smaller_target["data"]["target"]["high"] = 1.0
        configs[f"{stem}_sens_zeta1"] = smaller_target
    return configs


def theory_config() -> dict:
    """E0a-E0d identity and algorithm checks."""
    return {
        "e0_theory_checks": {
            "description": "E0a-E0d: classical reduction, Theorems 1, 2, 5, 6, 8, 13, Corollary 1, convolution identities, Monte Carlo of Algorithms 1, 2, 4, 5, 7",
            "experiment": {"task": "theory_checks", "id": "e0_theory_checks"},
            "checks": {
                "activations": ["tanh", "softplus", "silu"],
                "classical_reduction_qubits": [2, 3, 4],
                "problems": [
                    {"n_qubits": 2, "term_set": "tfim", "include_identity": True},
                    {
                        "n_qubits": 2,
                        "term_set": "heisenberg",
                        "include_identity": False,
                    },
                    {"n_qubits": 3, "term_set": "tfim", "include_identity": True},
                ],
                "convolution": {
                    "t1": 0.7,
                    "t2": 1.3,
                    "points": [-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0],
                },
                "monte_carlo": {
                    "n_qubits": 2,
                    "term_set": "tfim",
                    "single_shot_shots": 200000,
                    "shot_counts": [100, 1000, 10000],
                },
            },
        }
    }


E10_TEMPERATURES = {"fixed": [0.25, 0.5, 1.0, 2.0, 4.0], "sqrt_n_factors": [0.5, 1.0]}
E10_INPUTS = ["zero", "product", "ghz", "haar", "ginibre"]
E10_INIT_RANGES = {"a05": 0.5, "api": 3.141592653589793}


def _e10_config(
    term_set, n_qubits, range_name, temperatures, inputs, draws, data_check
):
    stem = f"e10_{term_set}_{n_qubits}q_{range_name}"
    return stem, {
        "description": f"E10 (extension X1): gradient variance, {term_set}, {n_qubits} qubits, theta ~ U[-a, a] with a = {E10_INIT_RANGES[range_name]:g}",
        "experiment": {"task": "gradient_variance", "id": stem},
        "gradient_variance": {
            "term_set": term_set,
            "include_identity": False,
            "n_qubits": n_qubits,
            "init_range": E10_INIT_RANGES[range_name],
            "temperatures": temperatures,
            "inputs": inputs,
            "parameter_positions": ["first", "middle", "last"],
            "draws": draws,
            "target_range": 2.0,
            "data_variance_check": data_check,
        },
    }


def extension_configs() -> dict:
    """E10, E11, E12: the user-requested trainability and contextuality extensions.

    Scope as pre-declared in LOG.md (2026-11-09T16:00Z amendment).
    """
    configs = {}
    for n_qubits in range(2, 11):
        for term_set in ("tfim", "heisenberg", "ising", "fcim", "tfim_global"):
            # At n = 2 the global Z^n string equals the ZZ coupling.
            if term_set == "tfim_global" and n_qubits < 3:
                continue
            for range_name in E10_INIT_RANGES:
                data_check = (
                    {
                        "temperature": 2.0,
                        "parameter_position": "first",
                        "n_states": 2000,
                    }
                    if range_name == "a05"
                    else None
                )
                stem, config = _e10_config(
                    term_set,
                    n_qubits,
                    range_name,
                    E10_TEMPERATURES,
                    E10_INPUTS,
                    200,
                    data_check,
                )
                configs[stem] = config
    for n_qubits in (11, 12):
        for term_set in ("tfim", "heisenberg"):
            stem, config = _e10_config(
                term_set,
                n_qubits,
                "a05",
                {"fixed": [2.0], "sqrt_n_factors": [1.0]},
                ["zero", "haar"],
                50,
                None,
            )
            configs[stem] = config
    for n_qubits in range(2, 10):
        for state_set in ("code", "haar"):
            stem = f"e11_tanh_{n_qubits}q_{state_set}"
            data = _squared_loss_data(
                n_qubits, "code_squared_loss", "tfim", True, "tanh"
            )
            if state_set == "haar":
                # Same number of training states as the code set: 2^(n+1) + 5.
                data["train_states"] = {
                    "set": "haar",
                    "n_states": 2 ** (n_qubits + 1) + 5,
                }
            configs[stem] = {
                "description": f"E11 (extension X1): Fig. 7a protocol (tanh, TFIM vs IM), {n_qubits} qubits, {state_set} training states",
                "experiment": {"task": "train", "id": stem},
                "data": data,
                "models": [
                    _model("quantum_tfim", "hamiltonian", "tfim", True, "tanh"),
                    _model("classical_im", "hamiltonian", "ising", True, "tanh"),
                ],
                "training": _training("squared", 0.1, 2000, 500),
            }
    temperature_grid = {"low": 0.03, "high": 100.0, "count": 25}
    configs["e12_invariance"] = {
        "description": "E12a (extension X3, amended): Kirby-Love flag of the support of phi(H) vs T for random noncontextual and contextual 2-local sets",
        "experiment": {"task": "contextuality_scan", "id": "e12_invariance"},
        "contextuality": {
            "temperatures": temperature_grid,
            "invariance": {
                "n_qubits": [3, 4],
                "sets_per_kind": 20,
                "sizes": [4, 5, 6, 7, 8],
                "locality": 2,
                "activations": ["tanh", "softplus", "silu"],
                "epsilon": 1e-8,
                "theta_range": 1.0,
            },
            "graded": None,
        },
    }
    for family in ("tfim", "heisenberg"):
        stem = f"e12_graded_{family}"
        configs[stem] = {
            "description": f"E12b (extension X3, amended): epsilon-support size, Kirby-Love flag and greedy contextual fraction of tanh(H/T) vs T, {family}, n = 2..5",
            "experiment": {"task": "contextuality_scan", "id": stem},
            "contextuality": {
                "temperatures": temperature_grid,
                "invariance": None,
                "graded": {
                    "families": [family],
                    "n_qubits": [2, 3, 4, 5],
                    "draws": 20,
                    "theta_range": 2.0,
                    "epsilons": [1e-2, 1e-3, 1e-4],
                    "activation": "tanh",
                    "graded_max_support": 300,
                },
            },
        }
    return configs


def _load(name):
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())


def e8_configs() -> dict:
    """E8 (claim C8): classical (basis-state) vs quantum (Haar) training data, same count."""
    configs = {}
    tasks = [(f"fig7_tanh_{n}q", n) for n in (2, 7)] + [
        (f"fig8_logistic_{n}q", n) for n in (3, 5, 7)
    ]
    for stem, n_qubits in tasks:
        base = _load(f"{stem}_original")
        for data_kind, states in (
            ("basis", {"set": "computational_basis"}),
            ("haar", {"set": "haar", "n_states": 2**n_qubits}),
        ):
            config = copy.deepcopy(base)
            config["description"] = (
                f"E8 (C8): {stem} original protocol, training states = {data_kind} ({2**n_qubits} states), validation 500 Haar states"
            )
            config["experiment"]["id"] = f"e8_{stem}_{data_kind}"
            config["data"]["train_states"] = states
            configs[f"e8_{stem}_{data_kind}"] = config
    return configs


def e9_configs() -> dict:
    """E9: learning-rate x temperature grid and finite-shot gradients."""
    configs = {}
    fig7_two = _load("fig7_tanh_2q_original")
    fig8_three = _load("fig8_logistic_3q_original")
    fig7_five = copy.deepcopy(fig7_two)
    fig7_five["data"]["n_qubits"] = 5
    fig8_five = _load("fig8_logistic_5q_original")
    for stem, base in (
        ("fig7_tanh_2q", fig7_two),
        ("fig7_tanh_5q", fig7_five),
        ("fig8_logistic_3q", fig8_three),
        ("fig8_logistic_5q", fig8_five),
    ):
        for learning_rate in (0.03, 0.1, 0.3):
            for temperature in (0.5, 1.0, 2.0, 4.0):
                name = f"e9_{stem}_eta{learning_rate:g}_T{temperature:g}"
                config = copy.deepcopy(base)
                config["description"] = (
                    f"E9a: {stem} original protocol with learning rate {learning_rate:g} and temperature {temperature:g}"
                )
                config["experiment"]["id"] = name
                config["physics"] = {"temperature": temperature}
                config["training"]["learning_rate"] = learning_rate
                configs[name] = config
    for shots in (10, 100, 1000):
        name = f"e9b_fig7_tanh_2q_shots{shots}"
        config = copy.deepcopy(fig7_two)
        config["description"] = (
            f"E9b: Fig. 7a protocol with finite-shot gradients, {shots} shots per state and parameter (Algorithms 1 and 2)"
        )
        config["experiment"]["id"] = name
        config["training"]["finite_shots"] = shots
        configs[name] = config
    return configs


E15_DRAWS = 3


def e15_term_sets(n_qubits: int) -> dict:
    """Term sets with J = 2n - 1 parameters for E15, random ones fixed by seed."""
    import numpy as np

    from fermi_dirac_neurons.lib import contextuality as cx

    sets = {
        "ising": ("ising", None),
        "ising_x0": ("ising_x0", None),
        "tfim": ("tfim", None),
    }
    size = 2 * n_qubits - 1
    for draw in range(E15_DRAWS):
        sets[f"ncrand{draw}"] = (
            "explicit",
            cx.random_noncontextual_two_local(
                n_qubits, size, np.random.default_rng([15, n_qubits, draw, 0])
            ),
        )
        sets[f"ctxrand{draw}"] = (
            "explicit",
            cx.random_contextual_labels(
                n_qubits, size, 2, np.random.default_rng([15, n_qubits, draw, 1])
            ),
        )
    return sets


def e15_configs() -> dict:
    """E15 (X5): contextual vs noncontextual term sets at T = 2 with J = 2n - 1."""
    configs = {}
    for n_qubits in (3, 5, 7):
        sets = e15_term_sets(n_qubits)
        models = []
        for name, (term_set, labels) in sets.items():
            model = _model(f"model_{name}", "hamiltonian", term_set, False, "tanh")
            if labels is not None:
                model["term_labels"] = labels
            models.append(model)
        fig7 = copy.deepcopy(_load("fig7_tanh_2q_original"))
        fig7["data"]["n_qubits"] = n_qubits
        fig8 = copy.deepcopy(_load(f"fig8_logistic_{n_qubits}q_original"))
        for stem, base in (
            (f"e15_learn_fig7_{n_qubits}q", fig7),
            (f"e15_learn_fig8_{n_qubits}q", fig8),
        ):
            config = copy.deepcopy(base)
            config["description"] = (
                f"E15 part 1: {stem.split('_')[2]} task, {n_qubits} qubits, term sets with J = {2 * n_qubits - 1} (commuting, noncontextual, contextual, random noncontextual and contextual)"
            )
            config["experiment"]["id"] = stem
            config["models"] = copy.deepcopy(models)
            configs[stem] = config
    for n_qubits in range(3, 11):
        sets = (
            e15_term_sets(n_qubits)
            if n_qubits <= 8
            else {k: (k, None) for k in ("ising", "ising_x0", "tfim")}
        )
        for name, (term_set, labels) in sets.items():
            if name.startswith(("ncrand", "ctxrand")) and n_qubits > 8:
                continue
            stem = f"e15_grad_{name}_{n_qubits}q"
            block = {
                "term_set": term_set,
                "include_identity": False,
                "n_qubits": n_qubits,
                "init_range": 0.5,
                "temperatures": {"fixed": [2.0], "sqrt_n_factors": []},
                "inputs": ["zero", "product", "haar"],
                "parameter_positions": ["first", "middle", "last"],
                "draws": 200,
                "target_range": 2.0,
                "data_variance_check": None,
            }
            if labels is not None:
                block["term_labels"] = labels
            configs[stem] = {
                "description": f"E15 part 2: gradient variance, term set {name}, {n_qubits} qubits, T = 2",
                "experiment": {"task": "gradient_variance", "id": stem},
                "gradient_variance": block,
            }
            if n_qubits <= 8:
                pauli_stem = f"e15_pauli_{name}_{n_qubits}q"
                pauli_block = {
                    "term_set": term_set,
                    "n_qubits": n_qubits,
                    "temperature": 2.0,
                    "activation": "tanh",
                    "theta_ranges": [0.5, 2.0],
                    "draws": 20,
                    "epsilons": [1e-2, 1e-3],
                }
                if labels is not None:
                    pauli_block["term_labels"] = labels
                configs[pauli_stem] = {
                    "description": f"E15 part 3: Pauli strings needed for tanh(H/T), term set {name}, {n_qubits} qubits, T = 2",
                    "experiment": {"task": "pauli_compressibility", "id": pauli_stem},
                    "pauli_compressibility": pauli_block,
                }
    return configs


def e16_configs() -> dict:
    """E16: Pauli-gap scaling (n = 9, 10), gradient decay of random sets (n = 9, 10), teacher-student."""
    configs = {}
    for n_qubits in (9, 10):
        for name, (term_set, labels) in e15_term_sets(n_qubits).items():
            pauli_block = {
                "term_set": term_set,
                "n_qubits": n_qubits,
                "temperature": 2.0,
                "activation": "tanh",
                "theta_ranges": [0.5, 2.0],
                "draws": 20,
                "epsilons": [1e-2, 1e-3],
            }
            if labels is not None:
                pauli_block["term_labels"] = labels
            stem = f"e16_pauli_{name}_{n_qubits}q"
            configs[stem] = {
                "description": f"E16a: Pauli strings needed for tanh(H/T), term set {name}, {n_qubits} qubits, T = 2",
                "experiment": {"task": "pauli_compressibility", "id": stem},
                "pauli_compressibility": pauli_block,
            }
            if labels is None:
                continue  # structured sets already cover n = 9, 10 in E15
            stem = f"e16_grad_{name}_{n_qubits}q"
            configs[stem] = {
                "description": f"E16c: gradient variance, term set {name}, {n_qubits} qubits, T = 2",
                "experiment": {"task": "gradient_variance", "id": stem},
                "gradient_variance": {
                    "term_set": term_set,
                    "term_labels": labels,
                    "include_identity": False,
                    "n_qubits": n_qubits,
                    "init_range": 0.5,
                    "temperatures": {"fixed": [2.0], "sqrt_n_factors": []},
                    "inputs": ["zero", "product", "haar"],
                    "parameter_positions": ["first", "middle", "last"],
                    "draws": 200,
                    "target_range": 2.0,
                    "data_variance_check": None,
                },
            }
    for n_qubits in (3, 5, 7):
        sets = e15_term_sets(n_qubits)
        students = []
        for name, (term_set, labels) in sets.items():
            model = _model(f"student_{name}", "hamiltonian", term_set, False, "tanh")
            if labels is not None:
                model["term_labels"] = labels
            students.append(model)
        base = copy.deepcopy(_load("fig7_tanh_2q_original"))
        base["data"]["n_qubits"] = n_qubits
        for teacher, (term_set, labels) in sets.items():
            stem = f"e16_teach_{teacher}_{n_qubits}q"
            config = copy.deepcopy(base)
            config["description"] = (
                f"E16b: Fig. 7a protocol with teacher term set {teacher} (tanh target, zeta ~ U[-2, 2]), all 9 term sets as students, {n_qubits} qubits"
            )
            config["experiment"]["id"] = stem
            target = {
                "term_set": term_set,
                "include_identity": False,
                "low": -2.0,
                "high": 2.0,
                "function": "tanh",
            }
            if labels is not None:
                target["term_labels"] = labels
            config["data"]["target"] = target
            config["models"] = copy.deepcopy(students)
            configs[stem] = config
    return configs


P4_LEARNING_SIZES = [(4, 2), (6, 3), (8, 4)]
P4_GRADIENT_SIZES = sorted(
    {(m, 2) for m in (2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20)}
    | {(m, 3) for m in (3, 4, 6, 8, 10, 12, 14, 16, 18, 20)}
    | {(m, m // 2) for m in (4, 6, 8, 10)}
)
P4_KERR_SIZES = sorted(
    {(m, n) for n in (2, 3) for m in (3, 4, 6, 8, 10, 12)} | {(8, 4), (10, 5)}
)
P4_INPUTS = ["fock_first", "haar_interferometer", "haar_full"]
P4_MODELS = [
    {"name": "photonic_tanh", "kind": "merlin_passive", "activation": "tanh"},
    {"name": "photonic_linear", "kind": "merlin_passive", "activation": "linear"},
    {"name": "photon_count_tanh", "kind": "photon_count"},
    {"name": "correlation_tanh", "kind": "correlation"},
]


def _fock_dimension(n_modes: int, n_photons: int) -> int:
    from math import comb

    return comb(n_modes + n_photons - 1, n_photons)


def _p4_learning(n_modes, n_photons, arm, epochs, eval_every, validation):
    d = _fock_dimension(n_modes, n_photons)
    train_states = (
        {"set": "haar_interferometer", "n_states": d}
        if arm == "haar"
        else {"set": "fock", "n_states": None}
    )
    return {
        "description": (
            f"Phase 4 MerLin adaptation (P4-L): photonic Fermi-Dirac neuron vs baselines, "
            f"N={n_photons} photons in m={n_modes} modes, {arm} training states, "
            "Fig. 7a protocol (squared loss, T = 2, eta = 0.1)"
        ),
        "experiment": {
            "task": "photonic_train",
            "id": f"p4_learn_m{n_modes}_n{n_photons}_{arm}",
        },
        "physics": {"temperature": 2.0},
        "photonic": {
            "n_modes": n_modes,
            "n_photons": n_photons,
            "target_range": 2.0,
            "train_states": train_states,
            "validation_states": {"set": "haar_interferometer", "n_states": validation},
            "evaluation": {
                "readouts": ["pnr", "threshold", "unbunched"],
                "shots": [100, 1000],
            },
        },
        "models": copy.deepcopy(P4_MODELS),
        "training": {
            "loss": "squared",
            "learning_rate": 0.1,
            "epochs": epochs,
            "eval_every": eval_every,
            "init_low": -0.5,
            "init_high": 0.5,
            "phase_init": [0.0, 6.283185307179586],
        },
    }


def _p4_gradient(n_modes, n_photons, draws):
    return {
        "description": f"Phase 4 E13 (X2): MerLin gradient variance, N={n_photons} photons in m={n_modes} modes",
        "experiment": {
            "task": "photonic_gradient_variance",
            "id": f"p4_grad_m{n_modes}_n{n_photons}",
        },
        "photonic_gradient_variance": {
            "n_modes": n_modes,
            "n_photons": n_photons,
            "draws": draws,
            "energy_ranges": [0.5, 3.141592653589793],
            "temperatures": [0.5, 2.0],
            "inputs": list(P4_INPUTS),
            "target_range": 2.0,
        },
    }


def _p4_kerr(n_modes, n_photons, draws):
    return {
        "description": f"Phase 4 E13-K (X2c, NOT MerLin): dense Fock-space gradient variance, passive H + cross-Kerr chain, N={n_photons} photons in m={n_modes} modes",
        "experiment": {
            "task": "kerr_gradient_variance",
            "id": f"p4_kerr_m{n_modes}_n{n_photons}",
        },
        "kerr_gradient_variance": {
            "n_modes": n_modes,
            "n_photons": n_photons,
            "draws": draws,
            "init_range": 0.5,
            "kerr": [0.0, 0.5, 2.0],
            "temperature": 2.0,
            "inputs": list(P4_INPUTS),
            "parameters": ["n0", "X0,1"],
        },
    }


def p4_configs() -> dict:
    """Phase 4 MerLin adaptation: learning (P4-L), trainability (E13) and the Kerr ablation (E13-K)."""
    configs = {}
    for n_modes, n_photons in P4_LEARNING_SIZES:
        for arm in ("haar", "fock"):
            config = _p4_learning(n_modes, n_photons, arm, 2000, 100, 500)
            configs[config["experiment"]["id"]] = config
    for n_modes, n_photons in P4_GRADIENT_SIZES:
        config = _p4_gradient(n_modes, n_photons, 200)
        configs[config["experiment"]["id"]] = config
    for n_modes, n_photons in P4_KERR_SIZES:
        config = _p4_kerr(n_modes, n_photons, 200)
        configs[config["experiment"]["id"]] = config
    smoke = _p4_learning(4, 2, "haar", 20, 10, 50)
    smoke["experiment"]["id"] = "p4_smoke"
    smoke["description"] = (
        "Phase 4 smoke: P4-L at (m, N) = (4, 2), 20 epochs, 50 validation states (not a result)"
    )
    configs["p4_smoke"] = smoke
    for name, builder in (("p4_grad_smoke", _p4_gradient), ("p4_kerr_smoke", _p4_kerr)):
        config = builder(3, 2, 5)
        config["experiment"]["id"] = name
        config["description"] = (
            f"Phase 4 smoke of {config['experiment']['task']} (m = 3, N = 2, 5 draws; not a result)"
        )
        configs[name] = config
    return configs


P5_TASKS = (
    [f"fig7_{act}_{n}q" for act in ("tanh", "softplus") for n in (2, 7)]
    + [f"fig8_logistic_{n}q" for n in range(2, 8)]
    + [f"fig9_{family}_{n}q" for family in ("tfim", "heis") for n in (2, 3)]
)
P5_LEARNING_RATES = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
P5_PHOTONIC_LEARNING_RATES = [0.003, 0.01, 0.03, 0.1]
P5_REGULARIZATION = [1e-6, 1e-4, 1e-2, 1.0]
P5_SELECTION_FRACTION = 0.2


def _p5_task_base(task: str) -> dict:
    """Original-protocol task with its B1 models plus the classical baselines B2 (and B3)."""
    config = copy.deepcopy(_load(f"{task}_original"))
    classical = _load(f"{task}_classical")
    names = {model["name"] for model in config["models"]}
    config["models"] += [
        copy.deepcopy(model)
        for model in classical["models"]
        if model["name"] not in names
    ]
    config["data"]["selection_fraction"] = P5_SELECTION_FRACTION
    config["training"]["on_nonfinite"] = "freeze"
    return config


def p5_configs() -> dict:
    """Phase 5 fair comparison: learning-rate selection, B4 reference, photonic selection.

    The stage-B configs (``p5_final_*``) depend on the stage-A results and
    are written by ``utils/select_p5.py``.
    """
    configs = {}
    for task in P5_TASKS:
        base = _p5_task_base(task)
        for rate in P5_LEARNING_RATES:
            stem = f"p5_sel_{task}_eta{rate:g}"
            config = copy.deepcopy(base)
            config["description"] = (
                f"Phase 5 stage A: {task} (original protocol) with B1, B2, B3 at learning rate {rate:g}, "
                "20 % of the training states held out for selection"
            )
            config["experiment"]["id"] = stem
            config["training"]["learning_rate"] = rate
            configs[stem] = config
        n_qubits = base["data"]["n_qubits"]
        stem = f"p5_ref_{task}"
        # Same data block (and physics, when overridden) as the task; no
        # training block, so the sweep tracks no epoch progress.
        reference = {
            key: copy.deepcopy(value)
            for key, value in base.items()
            if key not in ("description", "experiment", "models", "training")
        }
        reference |= {
            "description": f"Phase 5 B4 reference for {task}: ridge / logistic regression on Pauli expectations",
            "experiment": {"task": "linear_reference", "id": stem},
            "linear_reference": {
                "max_weights": [2] + ([n_qubits] if 2 < n_qubits <= 3 else []),
                "regularization": list(P5_REGULARIZATION),
            },
        }
        configs[stem] = reference
    for n_modes, n_photons in P4_LEARNING_SIZES:
        base = _p4_learning(n_modes, n_photons, "haar", 2000, 100, 500)
        base["photonic"]["selection_fraction"] = P5_SELECTION_FRACTION
        base["training"]["optimizer"] = "adam"
        base["training"]["on_nonfinite"] = "freeze"
        for rate in P5_PHOTONIC_LEARNING_RATES:
            stem = f"p5_phsel_m{n_modes}_n{n_photons}_lr{rate:g}"
            config = copy.deepcopy(base)
            config["description"] = (
                f"Phase 5 stage A (MerLin adaptation): four photonic models with Adam at learning rate {rate:g}, "
                f"N={n_photons} photons in m={n_modes} modes, interferometer data, 20 % selection split"
            )
            config["experiment"]["id"] = stem
            config["training"]["learning_rate"] = rate
            configs[stem] = config
    return configs


def main() -> None:
    """Write every config file and print the list."""
    configs = {
        **theory_config(),
        **figure7_configs(),
        **figure8_configs(),
        **figure9_configs(),
        **extension_configs(),
    }
    for name, config in configs.items():
        (CONFIG_DIR / f"{name}.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    # Later families copy the paper configs written above.
    configs |= {
        **e8_configs(),
        **e9_configs(),
        **e15_configs(),
        **e16_configs(),
        **p4_configs(),
        **p5_configs(),
    }
    for name, config in configs.items():
        (CONFIG_DIR / f"{name}.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {len(configs)} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
