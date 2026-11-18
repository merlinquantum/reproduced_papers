"""
QSVM Insurance Classification (80/10/10 train/val/test split, multi-seed)
===========================================================================

Behavior:
- Reads all *.pkl files from --data_path directory (20 seeds)
- For each seed: 80% train, 10% val, 10% test split
- Trains QSVM with quantum kernel on train set
- Evaluates on validation set (for hyperparameter selection)
- Evaluates on test set under the preserved preprocessing
- Records training and inference times
- Aggregates results across all seeds
- Uses a Qiskit CPU statevector by default; CUDA and MPI remain optional

Per-seed outputs:
- metrics_summary.csv (train/val/test metrics + timing)
- confusion_matrix_<split>.png
- qsvm_model_<qubits>qubits.pkl

Global outputs:
- metrics_over_seeds.csv (mean/std across seeds)
- timing_summary.csv (training/inference times across seeds)
- confusion_matrix_global_test.png
- summary_runs.csv

Usage:
mpirun -np 4 python scripts/qsvm_cuda_embeddings_insurance.py \
  --data_path /path/to/data_type6 \
  --output_dir Results/qsvm_insurance_80_10_10 \
  --qubits 2 \
  --seed 42
"""

import os

os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"

import argparse
import json
import re
import sys
import time
import warnings
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except ImportError:

    class _SerialComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def gather(self, value, root=0):
            return [value]

        def bcast(self, value, root=0):
            return value

    class _SerialMPI:
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()

from itertools import combinations, product

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated in 1\.9",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)

# Add parent directory to path for qve imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qve import (  # noqa: E402
    build_qsvm_qc,
    compute_cpu_kernel_entries,
    compute_zz_kernel_entries,
    data_partition,
    data_prepare_cv,
    data_to_operand,
    data_to_operand_reps,
    get_kernel_matrix,
    make_bsp,
    make_bsp_3dof,
    make_bsp_reps,
    make_zz_featuremap,
    normalize_kernel_cosine,
    normalize_kernel_frobenius,
    normalize_kernel_trace,
    normalize_train_and_cross_kernel_trace,
    operand_to_amp,
)
from qve.core import get_hybrid_kernel_matrix  # noqa: E402

# -----------------------------
# Helpers: discovery / loading
# -----------------------------


def extract_seed_from_name(path: str) -> Optional[int]:
    """Extract seed from filename (e.g. seed10) or parent dir (e.g. seed_4/)."""
    name = os.path.basename(path)
    m = re.search(r"seed_?(\d+)", name)
    if m:
        return int(m.group(1))
    parent = os.path.basename(os.path.dirname(path))
    m = re.search(r"seed_?(\d+)", parent)
    return int(m.group(1)) if m else None


def list_input_files(data_path: str, data_type_filter: str = None) -> list[str]:
    """List .pkl/.parquet files from a flat dir, seed_N/ subdirs, or a single file.

    Args:
        data_path: path to a single file or directory (flat or seed_N/ nested).
        data_type_filter: if set, only include files whose name contains this string.
            Supports '+'-separated multi-part filters where ALL parts must match
            (e.g. 'data_type5+cls_embedding' selects only CLS files for data_type5).
    """
    if os.path.isfile(data_path):
        return [data_path]

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"--data_path is neither a file nor a directory: {data_path}"
        )

    files = []
    filter_parts = data_type_filter.split("+") if data_type_filter else []
    for root, dirs, fnames in os.walk(data_path):
        dirs.sort()  # process seed_0, seed_1, ... in order
        for fn in fnames:
            if not (fn.endswith(".pkl") or fn.endswith(".parquet")):
                continue
            if filter_parts and not all(p in fn for p in filter_parts):
                continue
            files.append(os.path.join(root, fn))

    def sort_key(p: str):
        s = extract_seed_from_name(p)
        return (s if s is not None else 10**9, os.path.basename(p))

    files.sort(key=sort_key)
    return files


def load_data(data_path: str) -> tuple[pd.DataFrame, str]:
    """Load data from pickle or parquet format."""
    if data_path.endswith(".pkl") or data_path.endswith(".pickle"):
        return pd.read_pickle(data_path), "PICKLE"
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path), "PARQUET"
    return pd.read_csv(data_path), "CSV"


def parse_embedding(x):
    """Parse embedding; for PKL/PARQUET typically already arrays."""
    if not isinstance(x, str):
        return np.array(x, dtype=np.float32)

    s = x.replace("\n", " ").replace("[", " ").replace("]", " ")
    tokens = s.split()
    nums = []
    for t in tokens:
        if t == "...":
            continue
        try:
            nums.append(float(t))
        except ValueError:
            pass
    return np.array(nums, dtype=np.float32)


def process_embeddings(df: pd.DataFrame, file_format: str) -> pd.DataFrame:
    if "embedding" not in df.columns:
        raise ValueError("Column 'embedding' not found in the dataframe.")

    if file_format in ("PICKLE", "PARQUET"):
        df["emb_array"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        df["emb_array"] = df["embedding"].apply(parse_embedding)

    return df


def get_target_column(df: pd.DataFrame) -> str:
    for col in ["target", "label", "new_insurance_type", "insurance", "insurance_type"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No target column found. Expected one of: target, label, "
        "new_insurance_type, insurance, insurance_type"
    )


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], out_path: str, title: str
):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# QSVM Training / Evaluation
# -----------------------------


def run_qsvm_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    seed: int,
    device_id: int,
    comm_mpi,
    rank: int,
    size: int,
    class_names: list[str],
    use_hybrid: bool,
    alpha: float,
    classical_kernel: str,
    fix_leakage: bool = False,
    pi_angles: bool = False,
    balanced: bool = False,
    three_dof: bool = False,
    normalize_method: str = "trace",
    trace_protocol: str = "legacy_square_only",
    bandwidth: float = 1.0,
    circuit: str = "bsp",
    reps: int = 1,
    c_values: list = None,
    save_kernels: bool = False,
    output_dir: str = None,
    alpha_values: list = None,
    backend: str = "cpu",
):
    """
    80/10/10 train/val/test split for QSVM.
    Returns metrics dict with train/val/test results + timing.
    """
    if trace_protocol not in {"legacy_square_only", "train_trace"}:
        raise ValueError(f"Unsupported trace protocol: {trace_protocol}")

    minority_label = int(np.argmin(np.bincount(y)))

    # Create indices for tracking samples
    indices = np.arange(len(X))

    # Split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=seed
    )

    # Split temp: 50% val, 50% test (10% each of original)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    if rank == 0:
        print(
            f"  Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

    # Apply PCA and scaling
    pca_dim = n_qubits * 3 if three_dof else n_qubits
    svd_solver = "full" if backend == "cpu" else "auto"
    data_train, data_val = data_prepare_cv(
        pca_dim,
        X_train,
        X_val,
        fix_leakage=fix_leakage,
        pi_angles=pi_angles,
        svd_solver=svd_solver,
    )
    data_train_full, data_test = data_prepare_cv(
        pca_dim,
        X_train,
        X_test,
        fix_leakage=fix_leakage,
        pi_angles=pi_angles,
        svd_solver=svd_solver,
    )

    # Apply bandwidth scaling (quantum kernel gamma equivalent)
    if bandwidth != 1.0:
        data_train = data_train * bandwidth
        data_val = data_val * bandwidth
        data_train_full = data_train_full * bandwidth
        data_test = data_test * bandwidth
        print(f"  Bandwidth γ={bandwidth} applied to features")

    if rank == 0:
        print(f"  After PCA ({n_qubits} components) + Scaling")

    start_time = time.time()

    # Build index lists for kernel computation
    list_train = list(combinations(range(1, len(data_train) + 1), 2))
    list_val = list(product(range(1, len(data_val) + 1), range(1, len(data_train) + 1)))
    list_test = list(
        product(range(1, len(data_test) + 1), range(1, len(data_train_full) + 1))
    )

    # Partition for distributed processing
    list_train_partition = data_partition(list_train, size, rank)
    list_val_partition = data_partition(list_val, size, rank)
    list_test_partition = data_partition(list_test, size, rank)

    if rank == 0:
        print("  Building quantum circuit...")
    t0 = time.time()
    _use_slow_path = circuit == "zz"
    _use_reps_fast_path = (reps > 1) and (circuit != "zz")
    if circuit == "zz":
        feat_qc = make_zz_featuremap(n_qubits)
        if rank == 0:
            print(
                f"  Using ZZFeatureMap circuit (n_dim={n_qubits}) — assign_parameters path (no renew_operand)"
            )
    elif reps > 1:
        feat_qc = make_bsp_reps(n_qubits, reps)
        if rank == 0:
            print(
                f"  Using BSP reps={reps} circuit (n_dim={n_qubits}) — renew_operand_reps fast path"
            )
    elif three_dof:
        feat_qc = make_bsp_3dof(n_qubits)
    else:
        feat_qc = make_bsp(n_qubits)
    if backend == "cuda":
        import cupy as cp
        from cuquantum.cutensornet import CircuitToEinsum, Network, NetworkOptions

        kernel_qc = build_qsvm_qc(feat_qc, n_qubits, data_train[0], data_train[0])
        converter = CircuitToEinsum(kernel_qc, dtype="complex128", backend="cupy")
        a = str(0).zfill(n_qubits)
        exp, oper = converter.amplitude(a)
    circuit_time = time.time() - t0

    if rank == 0:
        print(f"  Computing quantum kernel on {backend.upper()}...")
    t0 = time.time()
    if backend == "cpu":
        list_train_full_partition = data_partition(
            list(combinations(range(1, len(data_train_full) + 1), 2)), size, rank
        )
        amp_train = compute_cpu_kernel_entries(
            feat_qc, n_qubits, list_train_partition, data_train, data_train
        )
        amp_val = compute_cpu_kernel_entries(
            feat_qc, n_qubits, list_val_partition, data_val, data_train
        )
        amp_train_full = compute_cpu_kernel_entries(
            feat_qc,
            n_qubits,
            list_train_full_partition,
            data_train_full,
            data_train_full,
        )
        amp_test = compute_cpu_kernel_entries(
            feat_qc, n_qubits, list_test_partition, data_test, data_train_full
        )
        operand_time = 0.0
        path_time = 0.0
        amplitude_time = time.time() - t0
        amp_data_train = comm_mpi.gather(np.asarray(amp_train), root=0)
        amp_data_val = comm_mpi.gather(np.asarray(amp_val), root=0)
        amp_data_train_full = comm_mpi.gather(np.asarray(amp_train_full), root=0)
        amp_data_test = comm_mpi.gather(np.asarray(amp_test), root=0)
    elif _use_slow_path:
        # ZZ: use assign_parameters path (renew_operand does not support ZZ)
        oper_train = compute_zz_kernel_entries(
            feat_qc, n_qubits, list_train_partition, data_train, data_train, device_id
        )
        oper_val = compute_zz_kernel_entries(
            feat_qc, n_qubits, list_val_partition, data_val, data_train, device_id
        )
        list_train_full_partition = data_partition(
            list(combinations(range(1, len(data_train_full) + 1), 2)), size, rank
        )
        oper_train_full = compute_zz_kernel_entries(
            feat_qc,
            n_qubits,
            list_train_full_partition,
            data_train_full,
            data_train_full,
            device_id,
        )
        oper_test = compute_zz_kernel_entries(
            feat_qc,
            n_qubits,
            list_test_partition,
            data_test,
            data_train_full,
            device_id,
        )
        operand_time = time.time() - t0

        # Gather from all ranks
        amp_data_train = comm_mpi.gather(cp.array(oper_train), root=0)
        amp_data_val = comm_mpi.gather(cp.array(oper_val), root=0)
        amp_data_train_full = comm_mpi.gather(cp.array(oper_train_full), root=0)
        amp_data_test = comm_mpi.gather(cp.array(oper_test), root=0)

        path_time = 0.0
        amplitude_time = operand_time
    elif _use_reps_fast_path:
        # BSP reps>1: use renew_operand_reps fast path (generalised renew_operand)
        oper_train = data_to_operand_reps(
            n_qubits, reps, oper, data_train, data_train, list_train_partition
        )
        oper_val = data_to_operand_reps(
            n_qubits, reps, oper, data_val, data_train, list_val_partition
        )
        # For test, use full training data
        oper_train_full = data_to_operand_reps(
            n_qubits,
            reps,
            oper,
            data_train_full,
            data_train_full,
            data_partition(
                list(combinations(range(1, len(data_train_full) + 1), 2)), size, rank
            ),
        )
        oper_test = data_to_operand_reps(
            n_qubits, reps, oper, data_test, data_train_full, list_test_partition
        )
        operand_time = time.time() - t0

        if rank == 0:
            print("  Setting up tensor network...")
        t0 = time.time()
        options = NetworkOptions(blocking="auto", device_id=device_id)
        network = Network(exp, *oper, options=options)
        path, info = network.contract_path()
        network.autotune(iterations=20)
        path_time = time.time() - t0

        if rank == 0:
            print("  Computing amplitudes...")
        t0 = time.time()

        len_train = len(oper_train)
        len_val = len(oper_val)
        len_train_full = len(oper_train_full)
        len_test = len(oper_test)

        oper_all = oper_train + oper_val + oper_train_full + oper_test
        amp_all = operand_to_amp(oper_all, network)

        idx = 0
        amp_train = amp_all[idx : idx + len_train]
        idx += len_train
        amp_val = amp_all[idx : idx + len_val]
        idx += len_val
        amp_train_full = amp_all[idx : idx + len_train_full]
        idx += len_train_full
        amp_test = amp_all[idx : idx + len_test]

        amp_data_train = comm_mpi.gather(cp.array(amp_train), root=0)
        amp_data_val = comm_mpi.gather(cp.array(amp_val), root=0)
        amp_data_train_full = comm_mpi.gather(cp.array(amp_train_full), root=0)
        amp_data_test = comm_mpi.gather(cp.array(amp_test), root=0)

        amplitude_time = time.time() - t0

    else:
        # BSP reps=1: use renew_operand fast path
        oper_train = data_to_operand(
            n_qubits, oper, data_train, data_train, list_train_partition
        )
        oper_val = data_to_operand(
            n_qubits, oper, data_val, data_train, list_val_partition
        )
        oper_train_full = data_to_operand(
            n_qubits,
            oper,
            data_train_full,
            data_train_full,
            data_partition(
                list(combinations(range(1, len(data_train_full) + 1), 2)), size, rank
            ),
        )
        oper_test = data_to_operand(
            n_qubits, oper, data_test, data_train_full, list_test_partition
        )
        operand_time = time.time() - t0

        if rank == 0:
            print("  Setting up tensor network...")
        t0 = time.time()
        options = NetworkOptions(blocking="auto", device_id=device_id)
        network = Network(exp, *oper, options=options)
        path, info = network.contract_path()
        network.autotune(iterations=20)
        path_time = time.time() - t0

        if rank == 0:
            print("  Computing amplitudes...")
        t0 = time.time()

        len_train = len(oper_train)
        len_val = len(oper_val)
        len_train_full = len(oper_train_full)
        len_test = len(oper_test)

        oper_all = oper_train + oper_val + oper_train_full + oper_test
        amp_all = operand_to_amp(oper_all, network)

        idx = 0
        amp_train = amp_all[idx : idx + len_train]
        idx += len_train
        amp_val = amp_all[idx : idx + len_val]
        idx += len_val
        amp_train_full = amp_all[idx : idx + len_train_full]
        idx += len_train_full
        amp_test = amp_all[idx : idx + len_test]

        amp_data_train = comm_mpi.gather(cp.array(amp_train), root=0)
        amp_data_val = comm_mpi.gather(cp.array(amp_val), root=0)
        amp_data_train_full = comm_mpi.gather(cp.array(amp_train_full), root=0)
        amp_data_test = comm_mpi.gather(cp.array(amp_test), root=0)

        amplitude_time = time.time() - t0

    results = None
    if rank == 0:
        print("  Building kernel matrices...")
        # Compute raw quantum kernels once, regardless of backend or alpha sweep
        K_quantum_train = get_kernel_matrix(
            data_train, data_train, amp_data_train, list_train, mode="train"
        )
        K_quantum_val = get_kernel_matrix(data_val, data_train, amp_data_val, list_val)
        K_quantum_train_full = get_kernel_matrix(
            data_train_full,
            data_train_full,
            amp_data_train_full,
            list(combinations(range(1, len(data_train_full) + 1), 2)),
            mode="train",
        )
        K_quantum_test = get_kernel_matrix(
            data_test, data_train_full, amp_data_test, list_test
        )

        cw = "balanced" if balanced else None

        if save_kernels:
            kern_dir = (
                os.path.join(output_dir, f"seed_{seed}", "kernels")
                if output_dir
                else os.path.join("kernels")
            )
            os.makedirs(kern_dir, exist_ok=True)
            np.save(os.path.join(kern_dir, "K_quantum_train.npy"), K_quantum_train)
            np.save(os.path.join(kern_dir, "K_quantum_val.npy"), K_quantum_val)
            np.save(
                os.path.join(kern_dir, "K_quantum_train_full.npy"), K_quantum_train_full
            )
            np.save(os.path.join(kern_dir, "K_quantum_test.npy"), K_quantum_test)
            np.save(os.path.join(kern_dir, "y_train.npy"), y_train)
            np.save(os.path.join(kern_dir, "y_val.npy"), y_val)
            np.save(os.path.join(kern_dir, "y_test.npy"), y_test)
            print(f"  Kernels saved to {kern_dir}")

        # ---------------------------------------------------------------
        # α-sweep path: compute K_classical ONCE, loop over α classically
        # ---------------------------------------------------------------
        if alpha_values is not None and use_hybrid:
            print(f"  α-sweep mode: {alpha_values}")
            print(f"  Computing classical {classical_kernel} kernel (once)...")
            if classical_kernel == "rbf":
                K_classical_train = rbf_kernel(data_train, data_train)
                K_classical_val = rbf_kernel(data_val, data_train)
                K_classical_train_full = rbf_kernel(data_train_full, data_train_full)
                K_classical_test = rbf_kernel(data_test, data_train_full)
            elif classical_kernel == "poly":
                K_classical_train = polynomial_kernel(data_train, data_train, degree=3)
                K_classical_val = polynomial_kernel(data_val, data_train, degree=3)
                K_classical_train_full = polynomial_kernel(
                    data_train_full, data_train_full, degree=3
                )
                K_classical_test = polynomial_kernel(
                    data_test, data_train_full, degree=3
                )
            else:  # linear
                K_classical_train = linear_kernel(data_train, data_train)
                K_classical_val = linear_kernel(data_val, data_train)
                K_classical_train_full = linear_kernel(data_train_full, data_train_full)
                K_classical_test = linear_kernel(data_test, data_train_full)

            mixing_normalize_method = normalize_method
            if normalize_method == "trace" and trace_protocol == "train_trace":
                K_classical_train, K_classical_val = (
                    normalize_train_and_cross_kernel_trace(
                        K_classical_train, K_classical_val
                    )
                )
                K_classical_train_full, K_classical_test = (
                    normalize_train_and_cross_kernel_trace(
                        K_classical_train_full, K_classical_test
                    )
                )
                K_quantum_train, K_quantum_val = normalize_train_and_cross_kernel_trace(
                    K_quantum_train, K_quantum_val
                )
                K_quantum_train_full, K_quantum_test = (
                    normalize_train_and_cross_kernel_trace(
                        K_quantum_train_full, K_quantum_test
                    )
                )
                mixing_normalize_method = "none"

            _c_values = c_values if c_values else [1.0]
            sweep_rows = []
            best_alpha_val_acc = -1.0
            best_alpha_row = None
            best_alpha_svc_full = None
            best_alpha_y_test_pred = None
            best_alpha_y_test_proba = None
            best_alpha_cm = None

            for a in alpha_values:
                print(f"\n  --- α={a:.4f} ---")
                # Combine kernels classically (no GPU needed)
                hybrid_K_train = get_hybrid_kernel_matrix(
                    K_classical_train,
                    K_quantum_train,
                    a,
                    normalize_method=mixing_normalize_method,
                )
                hybrid_K_val = get_hybrid_kernel_matrix(
                    K_classical_val,
                    K_quantum_val,
                    a,
                    normalize_method=mixing_normalize_method,
                )
                hybrid_K_train_full = get_hybrid_kernel_matrix(
                    K_classical_train_full,
                    K_quantum_train_full,
                    a,
                    normalize_method=mixing_normalize_method,
                )
                hybrid_K_test = get_hybrid_kernel_matrix(
                    K_classical_test,
                    K_quantum_test,
                    a,
                    normalize_method=mixing_normalize_method,
                )

                # C-grid search on val
                best_c_a, best_val_acc_a = _c_values[0], -1.0
                if len(_c_values) > 1:
                    print(f"    C-grid search over {_c_values}...")
                    for c in _c_values:
                        svc_tmp = SVC(
                            kernel="precomputed",
                            C=c,
                            probability=False,
                            random_state=seed,
                            class_weight=cw,
                        )
                        svc_tmp.fit(hybrid_K_train, y_train)
                        val_acc_c = svc_tmp.score(hybrid_K_val, y_val)
                        print(f"      C={c}: val_acc={val_acc_c:.4f}")
                        if val_acc_c > best_val_acc_a:
                            best_c_a, best_val_acc_a = c, val_acc_c
                    print(f"    Best C={best_c_a} (val_acc={best_val_acc_a:.4f})")
                else:
                    # Evaluate single C to get val accuracy
                    svc_tmp = SVC(
                        kernel="precomputed",
                        C=_c_values[0],
                        probability=False,
                        random_state=seed,
                        class_weight=cw,
                    )
                    svc_tmp.fit(hybrid_K_train, y_train)
                    best_val_acc_a = svc_tmp.score(hybrid_K_val, y_val)
                    best_c_a = _c_values[0]

                # Fit final train model
                t_train_start = time.time()
                svc_a = SVC(
                    kernel="precomputed",
                    C=best_c_a,
                    probability=True,
                    random_state=seed,
                    class_weight=cw,
                )
                svc_a.fit(hybrid_K_train, y_train)
                train_time_a = time.time() - t_train_start

                # Eval on train
                t_infer_train = time.time()
                y_train_pred_a = svc_a.predict(hybrid_K_train)
                y_train_proba_a = svc_a.predict_proba(hybrid_K_train)[:, 1]
                infer_train_time_a = time.time() - t_infer_train

                # Eval on val
                t_infer_val = time.time()
                y_val_pred_a = svc_a.predict(hybrid_K_val)
                y_val_proba_a = svc_a.predict_proba(hybrid_K_val)[:, 1]
                infer_val_time_a = time.time() - t_infer_val

                # Retrain on full train, eval on test
                print("    Retraining on full train for test evaluation...")
                svc_full_a = SVC(
                    kernel="precomputed",
                    C=best_c_a,
                    probability=True,
                    random_state=seed,
                    class_weight=cw,
                )
                svc_full_a.fit(hybrid_K_train_full, y_train)

                t_infer_test = time.time()
                y_test_pred_a = svc_full_a.predict(hybrid_K_test)
                y_test_proba_a = svc_full_a.predict_proba(hybrid_K_test)[:, 1]
                infer_test_time_a = time.time() - t_infer_test

                def compute_metrics_a(y_true, y_pred, y_proba, prefix):
                    return {
                        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
                        f"{prefix}_precision": float(precision_score(y_true, y_pred)),
                        f"{prefix}_recall": float(recall_score(y_true, y_pred)),
                        f"{prefix}_f1": float(f1_score(y_true, y_pred)),
                        f"{prefix}_minority_f1": float(
                            f1_score(
                                y_true,
                                y_pred,
                                pos_label=minority_label,
                                zero_division=0,
                            )
                        ),
                        f"{prefix}_auc": float(roc_auc_score(y_true, y_proba)),
                    }

                row = {"alpha": float(a)}
                row.update(
                    compute_metrics_a(y_train, y_train_pred_a, y_train_proba_a, "train")
                )
                row.update(compute_metrics_a(y_val, y_val_pred_a, y_val_proba_a, "val"))
                row.update(
                    compute_metrics_a(y_test, y_test_pred_a, y_test_proba_a, "test")
                )
                row["train_time_sec"] = train_time_a
                row["infer_train_time_sec"] = infer_train_time_a
                row["infer_val_time_sec"] = infer_val_time_a
                row["infer_test_time_sec"] = infer_test_time_a
                row["total_time_sec"] = time.time() - start_time
                row["circuit_time_sec"] = circuit_time
                row["operand_time_sec"] = operand_time
                row["path_time_sec"] = path_time
                row["amplitude_time_sec"] = amplitude_time
                row["best_c"] = best_c_a
                row["alpha_sweep"] = "yes"
                sweep_rows.append(row)

                if best_val_acc_a > best_alpha_val_acc:
                    best_alpha_val_acc = best_val_acc_a
                    best_alpha_row = dict(row)
                    best_alpha_svc_full = svc_full_a
                    best_alpha_y_test_pred = y_test_pred_a
                    best_alpha_y_test_proba = y_test_proba_a
                    best_alpha_cm = {
                        "train": confusion_matrix(y_train, y_train_pred_a),
                        "val": confusion_matrix(y_val, y_val_pred_a),
                        "test": confusion_matrix(y_test, y_test_pred_a),
                    }

            print(
                f"\n  Best α={best_alpha_row['alpha']:.4f} "
                f"(val_acc={best_alpha_val_acc:.4f})"
            )

            # Add a 'hybrid_best' row (copy of best alpha row, labelled)
            best_row_labelled = dict(best_alpha_row)
            best_row_labelled["alpha_sweep"] = "best"

            results = dict(best_alpha_row)
            results["alpha_sweep_rows"] = sweep_rows + [best_row_labelled]
            results["confusion_matrices"] = best_alpha_cm
            results["model"] = best_alpha_svc_full
            results["predictions"] = {
                "test": {
                    "y_true": y_test,
                    "y_pred": best_alpha_y_test_pred,
                    "y_proba": best_alpha_y_test_proba,
                }
            }
            results["split_data"] = {
                "train": {"X": X_train, "y": y_train, "indices": idx_train},
                "val": {"X": X_val, "y": y_val, "indices": idx_val},
                "test": {"X": X_test, "y": y_test, "indices": idx_test},
            }

        else:
            # ---------------------------------------------------------------
            # Single-α path (backward compatible)
            # ---------------------------------------------------------------
            kernel_train, kernel_valid = apply_hybrid_kernel(
                K_quantum_train,
                K_quantum_val,
                data_train,
                data_val,
                use_hybrid,
                alpha,
                classical_kernel,
                normalize_method=normalize_method,
                trace_protocol=trace_protocol,
            )
            kernel_train_full, kernel_test = apply_hybrid_kernel(
                K_quantum_train_full,
                K_quantum_test,
                data_train_full,
                data_test,
                use_hybrid,
                alpha,
                classical_kernel,
                normalize_method=normalize_method,
                trace_protocol=trace_protocol,
            )

            # Training
            print("  Training QSVM...")
            _c_values = c_values if c_values else [1.0]
            best_c, best_val_acc = _c_values[0], -1.0
            if len(_c_values) > 1:
                print(f"  C-grid search over {_c_values}...")
                for c in _c_values:
                    svc_tmp = SVC(
                        kernel="precomputed",
                        C=c,
                        probability=False,
                        random_state=seed,
                        class_weight=cw,
                    )
                    svc_tmp.fit(kernel_train, y_train)
                    val_acc = svc_tmp.score(kernel_valid, y_val)
                    print(f"    C={c}: val_acc={val_acc:.4f}")
                    if val_acc > best_val_acc:
                        best_c, best_val_acc = c, val_acc
                print(f"  Best C={best_c} (val_acc={best_val_acc:.4f})")
            t_train_start = time.time()
            svc = SVC(
                kernel="precomputed",
                C=best_c,
                probability=True,
                random_state=seed,
                class_weight=cw,
            )
            svc.fit(kernel_train, y_train)
            train_time = time.time() - t_train_start

            # Train set evaluation
            t_infer_train = time.time()
            y_train_pred = svc.predict(kernel_train)
            y_train_proba = svc.predict_proba(kernel_train)[:, 1]
            infer_train_time = time.time() - t_infer_train

            # Val set evaluation
            t_infer_val = time.time()
            y_val_pred = svc.predict(kernel_valid)
            y_val_proba = svc.predict_proba(kernel_valid)[:, 1]
            infer_val_time = time.time() - t_infer_val

            # Test set evaluation (retrain on full train for final test)
            print("  Retraining on full train for test evaluation...")
            svc_full = SVC(
                kernel="precomputed",
                C=best_c,
                probability=True,
                random_state=seed,
                class_weight=cw,
            )
            svc_full.fit(kernel_train_full, y_train)

            t_infer_test = time.time()
            y_test_pred = svc_full.predict(kernel_test)
            y_test_proba = svc_full.predict_proba(kernel_test)[:, 1]
            infer_test_time = time.time() - t_infer_test

            total_time = time.time() - start_time

            # Compute metrics
            def compute_metrics(y_true, y_pred, y_proba, prefix):
                return {
                    f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
                    f"{prefix}_precision": float(precision_score(y_true, y_pred)),
                    f"{prefix}_recall": float(recall_score(y_true, y_pred)),
                    f"{prefix}_f1": float(f1_score(y_true, y_pred)),
                    f"{prefix}_minority_f1": float(
                        f1_score(
                            y_true,
                            y_pred,
                            pos_label=minority_label,
                            zero_division=0,
                        )
                    ),
                    f"{prefix}_auc": float(roc_auc_score(y_true, y_proba)),
                }

            results = {}
            results.update(
                compute_metrics(y_train, y_train_pred, y_train_proba, "train")
            )
            results.update(compute_metrics(y_val, y_val_pred, y_val_proba, "val"))
            results.update(compute_metrics(y_test, y_test_pred, y_test_proba, "test"))

            results["train_time_sec"] = train_time
            results["infer_train_time_sec"] = infer_train_time
            results["infer_val_time_sec"] = infer_val_time
            results["infer_test_time_sec"] = infer_test_time
            results["total_time_sec"] = total_time
            results["circuit_time_sec"] = circuit_time
            results["operand_time_sec"] = operand_time
            results["path_time_sec"] = path_time
            results["amplitude_time_sec"] = amplitude_time
            results["best_c"] = best_c

            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred)
            cm_val = confusion_matrix(y_val, y_val_pred)
            cm_test = confusion_matrix(y_test, y_test_pred)

            results["confusion_matrices"] = {
                "train": cm_train,
                "val": cm_val,
                "test": cm_test,
            }
            results["model"] = svc_full
            results["predictions"] = {
                "test": {
                    "y_true": y_test,
                    "y_pred": y_test_pred,
                    "y_proba": y_test_proba,
                }
            }
            # Store split data for saving
            results["split_data"] = {
                "train": {"X": X_train, "y": y_train, "indices": idx_train},
                "val": {"X": X_val, "y": y_val, "indices": idx_val},
                "test": {"X": X_test, "y": y_test, "indices": idx_test},
            }

        preprocessing_protocol = (
            "train_only" if fix_leakage else "legacy_train_plus_heldout"
        )
        results["fix_leakage"] = fix_leakage
        results["preprocessing_protocol"] = preprocessing_protocol
        results["normalize_method"] = normalize_method
        results["trace_protocol"] = trace_protocol
        results["cross_kernel_scaled_with_train_trace"] = (
            normalize_method == "trace" and trace_protocol == "train_trace"
        )
        for row in results.get("alpha_sweep_rows", []):
            row["fix_leakage"] = fix_leakage
            row["preprocessing_protocol"] = preprocessing_protocol
            row["normalize_method"] = normalize_method
            row["trace_protocol"] = trace_protocol
            row["cross_kernel_scaled_with_train_trace"] = results[
                "cross_kernel_scaled_with_train_trace"
            ]

    return results


def save_seed_outputs(
    out_dir, results, seed, n_qubits, class_names, original_shape=None, data_path=None
):
    """Save per-seed outputs."""
    os.makedirs(out_dir, exist_ok=True)

    # Metrics CSV
    _skip_keys = {
        "confusion_matrices",
        "model",
        "predictions",
        "split_data",
        "alpha_sweep_rows",
    }
    if "alpha_sweep_rows" in results:
        # α-sweep: write all per-α rows (already includes the 'best' labelled row)
        pd.DataFrame(results["alpha_sweep_rows"]).to_csv(
            os.path.join(out_dir, "metrics_summary.csv"), index=False
        )
    else:
        metrics_dict = {k: v for k, v in results.items() if k not in _skip_keys}
        pd.DataFrame([metrics_dict]).to_csv(
            os.path.join(out_dir, "metrics_summary.csv"), index=False
        )

    # Confusion matrices
    for split, cm in results["confusion_matrices"].items():
        cm_path = os.path.join(out_dir, f"confusion_matrix_{split}.png")
        plot_confusion_matrix(
            cm,
            class_names,
            cm_path,
            f"Seed {seed} - {split.upper()} - {n_qubits} Qubits",
        )
        pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names],
        ).to_csv(os.path.join(out_dir, f"confusion_matrix_{split}.csv"))

    # Save model
    model_path = os.path.join(out_dir, f"qsvm_model_{n_qubits}qubits.pkl")
    joblib.dump(results["model"], model_path)

    # Save split data (samples used for training/val/test)
    if "split_data" in results:
        split_data = results["split_data"]

        # Save each split as pickle (preserves numpy arrays)
        for split_name, split_info in split_data.items():
            split_file = os.path.join(out_dir, f"samples_{split_name}.pkl")
            joblib.dump(
                {
                    "X": split_info["X"],
                    "y": split_info["y"],
                    "indices": split_info["indices"],
                },
                split_file,
            )

        # Save dataset info as JSON
        n_train = len(split_data["train"]["y"])
        n_val = len(split_data["val"]["y"])
        n_test = len(split_data["test"]["y"])
        n_total = n_train + n_val + n_test
        n_features_original = (
            original_shape[1] if original_shape else split_data["train"]["X"].shape[1]
        )
        n_features_after_pca = n_qubits

        from datetime import datetime

        dataset_info = {
            "experiment": {
                "type": "QSVM",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "qubits": n_qubits,
            },
            "data": {
                "source_path": os.path.abspath(data_path) if data_path else None,
                "source_filename": os.path.basename(data_path) if data_path else None,
            },
            "output": {"path": os.path.abspath(out_dir)},
            "seed": seed,
            "qubits": n_qubits,
            "fix_leakage": bool(results["fix_leakage"]),
            "preprocessing_protocol": results["preprocessing_protocol"],
            "normalize_method": results["normalize_method"],
            "trace_protocol": results["trace_protocol"],
            "cross_kernel_scaled_with_train_trace": bool(
                results["cross_kernel_scaled_with_train_trace"]
            ),
            "samples": {
                "total": n_total,
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "split_ratio": "80/10/10",
            },
            "features": {
                "original": n_features_original,
                "after_pca": n_features_after_pca,
            },
            "classes": class_names,
            "class_distribution": {
                "train": {
                    class_names[0]: int(np.sum(split_data["train"]["y"] == 0)),
                    class_names[1]: int(np.sum(split_data["train"]["y"] == 1)),
                },
                "val": {
                    class_names[0]: int(np.sum(split_data["val"]["y"] == 0)),
                    class_names[1]: int(np.sum(split_data["val"]["y"] == 1)),
                },
                "test": {
                    class_names[0]: int(np.sum(split_data["test"]["y"] == 0)),
                    class_names[1]: int(np.sum(split_data["test"]["y"] == 1)),
                },
            },
        }

        with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)


def apply_hybrid_kernel(
    kernel_train,
    kernel_valid,
    data_train,
    data_valid,
    use_hybrid=False,
    alpha=0.5,
    classical_kernel="rbf",
    normalize_method="trace",
    trace_protocol="legacy_square_only",
):
    """
    Apply hybrid kernel combination if enabled; optionally normalize quantum kernel.

    Args:
        kernel_train: Quantum kernel matrix for training (n_train, n_train)
        kernel_valid: Quantum kernel matrix for validation (n_valid, n_train)
        data_train: Training data for classical kernel (n_train, n_features)
        data_valid: Validation data for classical kernel (n_valid, n_features)
        use_hybrid: Whether to use hybrid kernel
        alpha: Mixing parameter (0=classical, 1=quantum)
        classical_kernel: Type of classical kernel ("rbf", "poly", "linear")
        normalize_method: Normalization applied to quantum kernel before combining
            ("trace", "frobenius", "cosine", "none")
        trace_protocol: Trace-scaling protocol. ``legacy_square_only`` preserves
            square-only normalization; ``train_trace`` scales each train/cross
            pair by the corresponding training trace.

    Returns:
        Tuple of (kernel_train, kernel_valid) — hybrid or normalized pure quantum
    """
    if trace_protocol not in {"legacy_square_only", "train_trace"}:
        raise ValueError(f"Unsupported trace protocol: {trace_protocol}")

    if not use_hybrid:
        # Pure QSVM: optionally normalize quantum kernel
        if normalize_method == "trace":
            if trace_protocol == "train_trace":
                kernel_train, kernel_valid = normalize_train_and_cross_kernel_trace(
                    kernel_train, kernel_valid
                )
            else:
                kernel_train = normalize_kernel_trace(kernel_train)
                kernel_valid = normalize_kernel_trace(kernel_valid)
        elif normalize_method == "frobenius":
            kernel_train = normalize_kernel_frobenius(kernel_train)
            kernel_valid = normalize_kernel_frobenius(kernel_valid)
        elif normalize_method == "cosine":
            kernel_train = normalize_kernel_cosine(kernel_train)
            kernel_valid = normalize_kernel_cosine(kernel_valid)
        return kernel_train, kernel_valid

    print(f"  Computing classical {classical_kernel} kernel...")

    # Compute classical kernel
    if classical_kernel == "rbf":
        classical_K_train = rbf_kernel(data_train, data_train)
        classical_K_valid = rbf_kernel(data_valid, data_train)
    elif classical_kernel == "poly":
        classical_K_train = polynomial_kernel(data_train, data_train, degree=3)
        classical_K_valid = polynomial_kernel(data_valid, data_train, degree=3)
    else:  # linear
        classical_K_train = linear_kernel(data_train, data_train)
        classical_K_valid = linear_kernel(data_valid, data_train)

    mixing_normalize_method = normalize_method
    if normalize_method == "trace" and trace_protocol == "train_trace":
        classical_K_train, classical_K_valid = normalize_train_and_cross_kernel_trace(
            classical_K_train, classical_K_valid
        )
        kernel_train, kernel_valid = normalize_train_and_cross_kernel_trace(
            kernel_train, kernel_valid
        )
        mixing_normalize_method = "none"

    # Combine kernels (normalize_method applied inside get_hybrid_kernel_matrix)
    print(f"  Combining with α={alpha:.2f} (Quantum) + {1 - alpha:.2f} (Classical)")
    hybrid_K_train = get_hybrid_kernel_matrix(
        classical_K_train,
        kernel_train,
        alpha,
        normalize_method=mixing_normalize_method,
    )
    hybrid_K_valid = get_hybrid_kernel_matrix(
        classical_K_valid,
        kernel_valid,
        alpha,
        normalize_method=mixing_normalize_method,
    )

    return hybrid_K_train, hybrid_K_valid


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="QSVM Insurance Classification")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to 20-seeds directory or a single data file",
    )
    parser.add_argument(
        "--data_type_filter",
        type=str,
        default=None,
        help="Select only files containing this string (e.g. 'data_type5')",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Max number of seed files to process (default: 10, 0=all)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--qubits", type=int, default=2, help="Number of qubits (PCA components)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (fallback if not in filename)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use (default: all)",
    )
    parser.add_argument(
        "--single_mode",
        action="store_true",
        help="Single file mode: save all outputs directly in output_dir (no seed/global subdirs)",
    )
    parser.add_argument(
        "--use_hybrid", action="store_true", help="Use hybrid quantum-classical kernel"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid kernel mixing parameter (0=classical, 1=quantum)",
    )
    parser.add_argument(
        "--alpha_values",
        type=str,
        default=None,
        help="Comma-separated α values to sweep (e.g. '0.1,0.3,0.5,0.7,0.9,1.0'). "
        "Requires --use_hybrid. Overrides --alpha. Best α selected by val accuracy.",
    )
    parser.add_argument(
        "--classical_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="Classical kernel type for hybrid approach",
    )
    parser.add_argument(
        "--fix_leakage",
        action="store_true",
        help="Fit MinMaxScaler on train only (default off — preserves legacy behavior)",
    )
    parser.add_argument(
        "--pi_angles",
        action="store_true",
        help="Scale features to [-pi, pi] instead of [-1, 1] (default off — preserves legacy behavior)",
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="trace",
        choices=["cosine", "trace", "frobenius", "none"],
        help="Kernel normalization method (default: trace)",
    )
    parser.add_argument(
        "--trace_protocol",
        choices=["legacy_square_only", "train_trace"],
        default="legacy_square_only",
        help="Trace scaling protocol (default: legacy_square_only)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use class_weight='balanced' in SVC to handle class imbalance (default off)",
    )
    parser.add_argument(
        "--three_dof",
        action="store_true",
        help="Use 3-DOF circuit (make_bsp_3dof): 3 distinct angles per qubit, PCA to 3*n_qubits (default off)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=1.0,
        help="Feature bandwidth multiplier applied after PCA+MinMaxScaler (default: 1.0)",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="bsp",
        choices=["bsp", "zz"],
        help="Feature map circuit: bsp (default, preserves existing behavior) or zz (ZZFeatureMap, Havlíček et al. 2019)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Data re-uploading repetitions for BSP circuit (default: 1). reps>1 uses assign_parameters path.",
    )
    parser.add_argument(
        "--c_values",
        type=str,
        default="1.0",
        help="Comma-separated C values for SVC grid search (default: 1.0)",
    )
    parser.add_argument(
        "--save_kernels",
        action="store_true",
        help="Save kernel matrices as .npy files for post-hoc analysis",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Quantum simulation backend (default: cpu; cuda requires qml-medimage[gpu])",
    )

    args = parser.parse_args()

    # Parse alpha_values sweep list (if provided)
    alpha_values_list = None
    if args.alpha_values is not None:
        alpha_values_list = [float(a) for a in args.alpha_values.split(",")]
        if not args.use_hybrid:
            raise ValueError("--alpha_values requires --use_hybrid")

    # MPI setup
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()
    device_id = 0
    if args.backend == "cuda":
        try:
            import cupy as cp
            from cupy.cuda.runtime import getDeviceCount
            from cuquantum.cutensornet import CircuitToEinsum, Network, NetworkOptions  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "CUDA backend unavailable. Install qml-medimage[gpu] on a supported "
                "Linux/NVIDIA system, or use --backend cpu."
            ) from exc
        device_count = getDeviceCount()
        if device_count < 1:
            raise RuntimeError("No CUDA GPU detected. Use --backend cpu.")
        device_id = rank % device_count
        cp.cuda.Device(device_id).use()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        files = list_input_files(args.data_path, data_type_filter=args.data_type_filter)
        if not files:
            raise FileNotFoundError(f"No data files found in: {args.data_path}")
        if args.num_seeds > 0:
            files = files[: args.num_seeds]
        print(f"Found {len(files)} seed files")
    else:
        files = None

    files = comm_mpi.bcast(files, root=0)

    global_rows = []
    global_test_preds = {"y_true": [], "y_pred": [], "y_proba": []}
    summary_rows = []
    global_class_names = None

    for fp in files:
        seed_from_file = extract_seed_from_name(fp)
        seed = seed_from_file if seed_from_file is not None else args.seed

        if rank == 0:
            print("\n" + "=" * 80)
            print(f"[FILE] {fp}")
            print(f"[SEED] {seed}")
            print(f"[QUBITS] {args.qubits}")
            print("=" * 80)

        try:
            if rank == 0:
                df, fmt = load_data(fp)
                df = process_embeddings(df, fmt)
                target_col = get_target_column(df)
                df = (
                    df[[target_col, "emb_array"]]
                    .dropna(subset=[target_col, "emb_array"])
                    .copy()
                )

                le = LabelEncoder()
                y = le.fit_transform(df[target_col].astype(str).values)
                class_names = le.classes_.tolist()

                if len(class_names) != 2:
                    raise ValueError(
                        f"Binary classification expected. Found {len(class_names)}: {class_names}"
                    )

                if global_class_names is None:
                    global_class_names = class_names
                elif class_names != global_class_names:
                    raise ValueError(
                        f"Class mismatch. Expected: {global_class_names}, Found: {class_names}"
                    )

                input_dtype = np.float64 if args.backend == "cpu" else np.float32
                X = (
                    np.stack(df["emb_array"].values)
                    .reshape(len(df), -1)
                    .astype(input_dtype)
                )
                original_shape = X.shape
                print(f"  Feature shape: {X.shape}")

                # Subsample if max_samples specified
                if args.max_samples is not None and args.max_samples < len(X):
                    print(f"  Subsampling from {len(X)} to {args.max_samples} samples")
                    indices = np.random.RandomState(seed).permutation(len(X))[
                        : args.max_samples
                    ]
                    X = X[indices]
                    y = y[indices]
            else:
                X, y, class_names, original_shape = None, None, None, None

            # Broadcast
            X = comm_mpi.bcast(X, root=0)
            y = comm_mpi.bcast(y, root=0)
            class_names = comm_mpi.bcast(class_names, root=0)
            original_shape = comm_mpi.bcast(original_shape, root=0)

            # Train
            results = run_qsvm_splits(
                X,
                y,
                args.qubits,
                seed,
                device_id,
                comm_mpi,
                rank,
                size,
                class_names,
                args.use_hybrid,
                args.alpha,
                args.classical_kernel,
                fix_leakage=args.fix_leakage,
                pi_angles=args.pi_angles,
                balanced=args.balanced,
                three_dof=args.three_dof,
                normalize_method=args.normalize_method,
                trace_protocol=args.trace_protocol,
                bandwidth=args.bandwidth,
                circuit=args.circuit,
                reps=args.reps,
                c_values=[float(c) for c in args.c_values.split(",")],
                save_kernels=args.save_kernels,
                output_dir=args.output_dir,
                alpha_values=alpha_values_list,
                backend=args.backend,
            )

            if rank == 0:
                if args.single_mode:
                    # Single mode: save directly in output_dir (no subdirs)
                    save_seed_outputs(
                        args.output_dir,
                        results,
                        seed,
                        args.qubits,
                        class_names,
                        original_shape,
                        data_path=fp,
                    )
                else:
                    # Multi-seed mode: save in seed subdirectory
                    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
                    save_seed_outputs(
                        seed_dir,
                        results,
                        seed,
                        args.qubits,
                        class_names,
                        original_shape,
                        data_path=fp,
                    )

                    # Collect for global aggregation
                    _global_skip = {
                        "confusion_matrices",
                        "model",
                        "predictions",
                        "split_data",
                        "alpha_sweep_rows",
                    }
                    metrics_row = {
                        k: v for k, v in results.items() if k not in _global_skip
                    }
                    metrics_row["seed"] = seed
                    global_rows.append(metrics_row)

                    # Collect test predictions
                    global_test_preds["y_true"].extend(
                        results["predictions"]["test"]["y_true"].tolist()
                    )
                    global_test_preds["y_pred"].extend(
                        results["predictions"]["test"]["y_pred"].tolist()
                    )
                    global_test_preds["y_proba"].extend(
                        results["predictions"]["test"]["y_proba"].tolist()
                    )

                    summary_rows.append(
                        {
                            "file": fp,
                            "seed_from_filename": seed_from_file,
                            "seed_used": seed,
                            "samples_total": len(X),
                            "qubits": args.qubits,
                            "status": "OK",
                        }
                    )

        except Exception as e:
            if rank == 0:
                print(f"[ERROR] {repr(e)}")
                if not args.single_mode:
                    summary_rows.append(
                        {
                            "file": fp,
                            "seed_from_filename": seed_from_file,
                            "seed_used": seed,
                            "samples_total": None,
                            "qubits": args.qubits,
                            "status": f"FAIL: {repr(e)}",
                        }
                    )

    # Global aggregation (skip in single mode)
    if rank == 0:
        if args.single_mode:
            print("\n" + "=" * 80)
            print("DONE (single mode)")
            print(f"Outputs written to: {args.output_dir}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("AGGREGATING GLOBAL RESULTS")
            print("=" * 80)

            global_root = os.path.join(args.output_dir, "global")
            os.makedirs(global_root, exist_ok=True)

            pd.DataFrame(summary_rows).to_csv(
                os.path.join(args.output_dir, "summary_runs.csv"), index=False
            )

            if global_rows:
                all_metrics = pd.DataFrame(global_rows)
                # Only aggregate numeric columns (exclude seed and non-numeric columns like classical_kernel)
                numeric_cols = all_metrics.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                metric_cols = [c for c in numeric_cols if c != "seed"]
                agg = all_metrics[metric_cols].agg(["mean", "std"]).T
                agg.to_csv(os.path.join(global_root, "metrics_over_seeds.csv"))

                # Timing summary
                timing_cols = [c for c in metric_cols if "time" in c]
                timing = all_metrics[["seed"] + timing_cols].copy()
                timing.to_csv(
                    os.path.join(global_root, "timing_summary.csv"), index=False
                )

                # Global test confusion matrix
                yt = np.array(global_test_preds["y_true"], dtype=int)
                yp = np.array(global_test_preds["y_pred"], dtype=int)
                cm_global = confusion_matrix(yt, yp)
                plot_confusion_matrix(
                    cm_global,
                    global_class_names,
                    os.path.join(global_root, "confusion_matrix_global_test.png"),
                    f"GLOBAL - Test Set - {args.qubits} Qubits",
                )
                pd.DataFrame(
                    cm_global,
                    index=[f"true_{c}" for c in global_class_names],
                    columns=[f"pred_{c}" for c in global_class_names],
                ).to_csv(os.path.join(global_root, "confusion_matrix_global_test.csv"))

            print("\n" + "=" * 80)
            print("DONE")
            print(f"Outputs written to: {args.output_dir}")
            print("=" * 80)


if __name__ == "__main__":
    main()
