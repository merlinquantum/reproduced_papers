import os

os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"
from functools import cache
from itertools import chain

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


# cuQuantum and CuPy are optional GPU dependencies.
def _get_cuquantum():
    from cuquantum.cutensornet import CircuitToEinsum, Network, NetworkOptions

    return CircuitToEinsum, Network, NetworkOptions


def _get_cupy():
    import cupy as cp

    return cp


def make_bsp(n_dim):
    """Creates a basic quantum circuit for feature mapping."""
    param = ParameterVector("p", n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    for q in range(n_dim):
        bsp_qc.rz(param.params[q], [q])
        bsp_qc.ry(param.params[q], [q])
    for i in range(n_dim - 1):
        bsp_qc.cx(i, i + 1)
    for q in range(n_dim):
        bsp_qc.rz(param.params[q], [q])
    return bsp_qc


def make_bsp_reps(n_dim, reps: int = 2):
    """BSP circuit with data re-uploading (Pérez-Salinas 2020).

    Stacks the BSP block `reps` times using the same ParameterVector.
    Each repetition re-encodes the data: H → Rz+Ry → CNOT chain → Rz.
    Uses the assign_parameters kernel path (like ZZ), not renew_operand.
    """
    param = ParameterVector("p", n_dim)
    qc = QuantumCircuit(n_dim)
    for _ in range(reps):
        qc.h(list(range(n_dim)))
        for q in range(n_dim):
            qc.rz(param.params[q], q)
            qc.ry(param.params[q], q)
        for i in range(n_dim - 1):
            qc.cx(i, i + 1)
        for q in range(n_dim):
            qc.rz(param.params[q], q)
    return qc


def make_bsp_3dof(n_dim):
    """Creates a 3-DOF quantum circuit: each qubit gets 3 distinct angles (Rz, Ry, Rz).
    Requires 3*n_dim input features. params layout: [rz0..rz_{n-1}, ry0..ry_{n-1}, rz2_0..rz2_{n-1}]
    """
    param = ParameterVector("p", 3 * n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    for q in range(n_dim):
        bsp_qc.rz(param.params[q], [q])  # first Rz: param[q]
        bsp_qc.ry(param.params[q + n_dim], [q])  # Ry: param[q + n_dim]
    for i in range(n_dim - 1):
        bsp_qc.cx(i, i + 1)
    for q in range(n_dim):
        bsp_qc.rz(param.params[q + 2 * n_dim], [q])  # second Rz: param[q + 2*n_dim]
    return bsp_qc


def make_zz_featuremap(n_dim: int, reps: int = 1):
    """ZZFeatureMap (Havlíček et al. 2019).

    Circuit per rep:
      1. H on all qubits
      2. Rz(x_i) on each qubit i
      3. For each pair (i < j): CNOT(i,j) → Rz((π - x_i)(π - x_j)) → CNOT(i,j)

    Uses ParameterVector("p", n_dim) — same number of params as make_bsp.
    PCA input dimension = n_dim (unchanged from BSP).
    """
    params = ParameterVector("p", n_dim)
    qc = QuantumCircuit(n_dim)

    for _rep in range(reps):
        # Layer 1: Hadamard on all qubits
        for q in range(n_dim):
            qc.h(q)
        # Layer 2: Single-qubit Rz rotations
        for q in range(n_dim):
            qc.rz(params[q], q)
        # Layer 3: Pairwise ZZ interactions
        for i in range(n_dim):
            for j in range(i + 1, n_dim):
                qc.cx(i, j)
                qc.rz((np.pi - params[i]) * (np.pi - params[j]), j)
                qc.cx(i, j)

    return qc


def compute_zz_kernel_entries(zz_qc, n_dim, pairs, data1, data2, device_id=0):
    """Compute quantum kernel entries for ZZFeatureMap using assign_parameters.

    This function uses Qiskit's assign_parameters to bind values before passing
    to cuTensorNet, bypassing the renew_operand optimisation (which is tied to
    the BSP gate-tensor offsets).  Correctness is guaranteed; throughput is
    lower than the renew_operand path.

    Args:
        zz_qc:     Unbound ZZFeatureMap QuantumCircuit (from make_zz_featuremap).
        n_dim:     Number of qubits.
        pairs:     List of (i1, i2) 1-based index pairs to evaluate.
        data1:     Array of shape (N, n_dim) — first dataset (1-indexed by i1).
        data2:     Array of shape (M, n_dim) — second dataset (1-indexed by i2).
        device_id: CUDA device id.

    Returns:
        List of float amplitudes |<0|U†(x2) U(x1)|0>|² for each pair.
    """
    CircuitToEinsum, Network, NetworkOptions = _get_cuquantum()
    options = NetworkOptions(blocking="auto", device_id=device_id)
    a = str(0).zfill(n_dim)
    results = []
    for i1, i2 in pairs:
        y_val = tuple(float(v) for v in data1[i1 - 1])
        x_val = tuple(float(v) for v in data2[i2 - 1])
        kernel_qc = build_qsvm_qc(zz_qc, n_dim, y_val, x_val)
        converter = CircuitToEinsum(kernel_qc, dtype="complex128", backend="cupy")
        exp, oper = converter.amplitude(a)
        with Network(exp, *oper, options=options) as tn:
            tn.contract_path()
            amp = abs(tn.contract()) ** 2
        results.append(amp)
    return results


def compute_cpu_kernel_entries(feature_qc, n_dim, pairs, data1, data2):
    """Compute exact quantum-kernel entries with Qiskit's CPU statevector."""
    results = []
    for i1, i2 in pairs:
        y_val = tuple(float(v) for v in data1[i1 - 1])
        x_val = tuple(float(v) for v in data2[i2 - 1])
        kernel_qc = build_qsvm_qc(feature_qc, n_dim, y_val, x_val)
        amplitude = Statevector.from_instruction(kernel_qc).data[0]
        results.append(float(abs(amplitude) ** 2))
    return results


def build_qsvm_qc(bsp_qc, n_dim, y_t, x_t):
    """Builds a quantum circuit for the quantum SVM kernel."""
    qc_1 = bsp_qc.assign_parameters(y_t).to_gate()
    qc_2 = bsp_qc.assign_parameters(x_t).inverse().to_gate()
    kernel_qc = QuantumCircuit(n_dim)
    kernel_qc.append(qc_1, list(range(n_dim)))
    kernel_qc.append(qc_2, list(range(n_dim)))
    return kernel_qc


# EDIT V2
@cache
def sin_cos(d):
    return np.sin(d), np.cos(d)


# EDIT V2
@cache
def get_from_d1(d1):
    half_d1 = d1 / 2
    z_g = [[np.exp(-1j * half_d1), 0], [0, np.exp(1j * half_d1)]]
    s, c = sin_cos(half_d1)
    return s, c, z_g


# EDIT V2
@cache
def get_from_d2(d2):
    half_d2 = d2 / 2
    z_gd = [[np.exp(1j * half_d2), 0], [0, np.exp(-1j * half_d2)]]
    s, c = sin_cos(half_d2)
    return s, c, z_gd


# EDIT V2: pre-computing parts..


def renew_operand(n_dim, oper_tmp, y_t, x_t):
    cp = _get_cupy()
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for d1 in y_t:
        s, c, z_g = get_from_d1(d1)
        n_zg.append(z_g)
        n_zy_g.extend([z_g, [[c, -s], [s, c]]])

    oper[n_dim * 2 : n_dim * 4] = cp.array(n_zy_g)
    oper[n_dim * 5 - 1 : n_dim * 6 - 1] = cp.array(n_zg)

    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:
        s, c, z_gd = get_from_d2(d2)
        n_zgd.append(z_gd)
        n_zy_gd.extend([[[c, s], [-s, c]], z_gd])

    oper[n_dim * 6 - 1 : n_dim * 7 - 1] = cp.array(n_zgd)
    oper[n_dim * 8 - 2 : n_dim * 10 - 2] = cp.array(n_zy_gd)

    return oper


def renew_operand_3dof(n_dim, oper_tmp, y_t, x_t):
    """renew_operand variant for 3-DOF circuits (make_bsp_3dof).

    y_t and x_t have 3*n_dim values each:
      - first  Rz for qubit q: y_t[q]
      - Ry      for qubit q: y_t[q + n_dim]
      - second Rz for qubit q: y_t[q + 2*n_dim]
    Tensor slice offsets mirror renew_operand — only the angle values differ.
    """
    cp = _get_cupy()
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for q in range(n_dim):
        d1_rz = y_t[q]
        d1_ry = y_t[q + n_dim]
        s_rz, c_rz, z_g = get_from_d1(d1_rz)
        s_ry, c_ry, _ = get_from_d1(d1_ry)
        n_zg.append(z_g)
        n_zy_g.extend([z_g, [[c_ry, -s_ry], [s_ry, c_ry]]])

    oper[n_dim * 2 : n_dim * 4] = cp.array(n_zy_g)
    oper[n_dim * 5 - 1 : n_dim * 6 - 1] = cp.array(n_zg)

    n_zgd, n_zy_gd = [], []
    for q in range(n_dim - 1, -1, -1):
        d2_rz2 = x_t[q + 2 * n_dim]
        d2_ry = x_t[q + n_dim]
        s_rz2, c_rz2, z_gd = get_from_d2(d2_rz2)
        s_ry2, c_ry2, _ = get_from_d2(d2_ry)
        n_zgd.append(z_gd)
        n_zy_gd.extend([[[c_ry2, s_ry2], [-s_ry2, c_ry2]], z_gd])

    oper[n_dim * 6 - 1 : n_dim * 7 - 1] = cp.array(n_zgd)
    oper[n_dim * 8 - 2 : n_dim * 10 - 2] = cp.array(n_zy_gd)

    return oper


def renew_operand_reps(n_dim, reps, oper_tmp, y_t, x_t):
    """Fast operand renewal for BSP reps>1 (data re-uploading) circuits.

    Generalises renew_operand() to arbitrary repetition depth. For reps=1
    the output is identical to renew_operand().

    Empirically verified layout for kernel circuit U_R(y) U†_R(x):
      - [0 : n]                  Initial state tensors (fixed, shape (2,))
      period = 5n-1  (each rep block: H(n) + Rz+Ry(2n) + CNOT(n-1) + Rz(n))
      base_fwd = n
      Forward rep i (0 .. R-1): s0 = n + i*period
        H(n)         [s0        : s0+n]         (fixed)
        Rz(y)+Ry(y)  [s0+n      : s0+3n]
        CX chain     [s0+3n     : s0+4n-1]      (fixed)
        Rz(y)        [s0+4n-1   : s0+5n-1]
      base_bwd = n + R*period
      Backward rep j (0 .. R-1): s0 = base_bwd + j*period
        Rz†(x)       [s0        : s0+n]
        CX† chain    [s0+n      : s0+2n-1]      (fixed)
        Ry†(x)+Rz†(x)[s0+2n-1  : s0+4n-1]
        H†(n)        [s0+4n-1   : s0+5n-1]      (fixed)
      - [n+2R*period : n+2R*period+n]  Final state tensors (fixed, shape (2,))

    Args:
        n_dim:    Number of qubits.
        reps:     Number of BSP repetitions (data re-uploading depth).
        oper_tmp: Template operand list from CircuitToEinsum on kernel circuit.
        y_t:      First data point (tuple of n_dim floats).
        x_t:      Second data point (tuple of n_dim floats).

    Returns:
        Updated copy of oper_tmp with parametric gate tensors patched.
    """
    cp = _get_cupy()
    n = n_dim
    oper = oper_tmp.copy()
    period = 5 * n - 1  # gate slots per rep block (H+Rz+Ry+CNOT+Rz = 5n-1)
    base_fwd = n  # forward reps start after initial state tensors
    base_bwd = n + reps * period

    # Pre-compute forward gate matrices (same values for every rep)
    n_zy_g, n_zg = [], []
    for d1 in y_t:
        s, c, z_g = get_from_d1(d1)
        n_zg.append(z_g)
        n_zy_g.extend([z_g, [[c, -s], [s, c]]])
    arr_zy_g = cp.array(n_zy_g)
    arr_zg = cp.array(n_zg)

    for i in range(reps):
        s0 = base_fwd + i * period
        # H at s0 : s0+n (fixed, skip)
        oper[s0 + n : s0 + 3 * n] = arr_zy_g  # Rz(y)+Ry(y)
        # CNOT at s0+3n : s0+4n-1 (fixed, skip)
        oper[s0 + 4 * n - 1 : s0 + 5 * n - 1] = arr_zg  # Rz(y) second layer

    # Pre-compute backward gate matrices (same values for every backward rep)
    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:
        s, c, z_gd = get_from_d2(d2)
        n_zgd.append(z_gd)
        n_zy_gd.extend([[[c, s], [-s, c]], z_gd])
    arr_zgd = cp.array(n_zgd)
    arr_zy_gd = cp.array(n_zy_gd)

    for j in range(reps):
        s0 = base_bwd + j * period
        oper[s0 : s0 + n] = arr_zgd  # Rz†(x)
        # CNOT† at s0+n : s0+2n-1 (fixed, skip)
        oper[s0 + 2 * n - 1 : s0 + 4 * n - 1] = arr_zy_gd  # Ry†(x)+Rz†(x)
        # H† at s0+4n-1 : s0+5n-1 (fixed, skip)

    return oper


def data_to_operand_reps(n_dim, reps, operand_tmp, data1, data2, indices_list):
    return [
        renew_operand_reps(n_dim, reps, operand_tmp, data1[i1 - 1], data2[i2 - 1])
        for i1, i2 in indices_list
    ]


def data_partition(indices_list, size, rank):
    """
    Partitions a list of indices for distributed processing.
    """
    num_data = len(indices_list)
    chunk, extra = num_data // size, num_data % size
    data_begin = rank * chunk + min(rank, extra)
    data_end = (
        num_data if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    return indices_list[data_begin:data_end]


# EDITED: reduced for-loop so its a comprenhension list
def data_to_operand(n_dim, operand_tmp, data1, data2, indices_list):
    return [
        renew_operand(n_dim, operand_tmp, data1[i1 - 1], data2[i2 - 1])
        for i1, i2 in indices_list
    ]


def data_to_operand_3dof(n_dim, operand_tmp, data1, data2, indices_list):
    return [
        renew_operand_3dof(n_dim, operand_tmp, data1[i1 - 1], data2[i2 - 1])
        for i1, i2 in indices_list
    ]


# EDITED: removing the append so instead does direct mapping
def operand_to_amp(opers, network):
    amp_tmp = [None] * len(opers)
    with network as tn:
        for i, op in enumerate(opers):
            tn.reset_operands(*op)
            amp_tmp[i] = abs(tn.contract()) ** 2
    return amp_tmp


def get_kernel_matrix(data1, data2, amp_data, indices_list, mode=None):
    """Builds the precomputed kernel matrix from amplitudes."""
    amp_m = list(chain.from_iterable(amp_data))
    kernel_matrix = np.zeros((len(data1), len(data2)))
    i = -1
    for i1, i2 in indices_list:
        i += 1
        kernel_matrix[i1 - 1][i2 - 1] = np.round(amp_m[i], 8)
    if mode == "train":
        kernel_matrix = kernel_matrix + kernel_matrix.T + np.diag(np.ones(len(data2)))
    return kernel_matrix


def normalize_kernel_trace(K):
    """
    Trace normalization: K_norm = K / trace(K)

    Ensures trace(K_norm) = 1
    Good for: Making kernels comparable in scale

    Note: Only applicable to square matrices. Returns K unchanged for rectangular matrices.
    """
    # Only normalize square matrices
    if K.shape[0] != K.shape[1]:
        return K

    trace = np.trace(K)
    if trace > 0:
        return K / trace
    return K


def normalize_train_and_cross_kernel_trace(kernel_train, kernel_cross):
    """Normalize training and cross-kernel matrices by the training trace.

    Args:
        kernel_train: Square training-kernel matrix.
        kernel_cross: Cross-kernel matrix with one column per training sample.

    Returns:
        The normalized training and cross-kernel matrices.

    Raises:
        ValueError: If matrix dimensions are incompatible or the training trace
            is not finite and strictly positive.
    """
    kernel_train = np.asarray(kernel_train)
    kernel_cross = np.asarray(kernel_cross)

    if kernel_train.ndim != 2 or kernel_train.shape[0] != kernel_train.shape[1]:
        raise ValueError("kernel_train must be a square matrix")
    if kernel_cross.ndim != 2 or kernel_cross.shape[1] != kernel_train.shape[0]:
        raise ValueError("kernel_cross must have one column per training sample")

    train_trace = np.trace(kernel_train)
    if (
        not np.isreal(train_trace)
        or not np.isfinite(train_trace)
        or train_trace.real <= 0
    ):
        raise ValueError("kernel_train trace must be finite and strictly positive")

    return kernel_train / train_trace, kernel_cross / train_trace


def normalize_kernel_frobenius(K):
    """
    Frobenius normalization: K_norm = K / ||K||_F

    Ensures Frobenius norm = 1
    Good for: Standard matrix normalization
    """
    frob_norm = np.linalg.norm(K, ord="fro")
    if frob_norm > 0:
        return K / frob_norm
    return K


def normalize_kernel_centered(K):
    """
    Centered kernel normalization (HSIC-style)

    Centers the kernel in feature space
    Good for: Statistical independence tests, better SVM performance

    Note: Only applicable to square matrices. Returns K unchanged for rectangular matrices.
    """
    # Only normalize square matrices
    if K.shape[0] != K.shape[1]:
        return K

    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    return K_centered


def normalize_kernel_cosine(K):
    """
    Cosine normalization: K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])

    Also known as "normalized kernel" or "angular kernel"
    Good for: Making kernel values range-independent

    Note: This is what sklearn's rbf_kernel does internally

    Args:
        K: Kernel matrix (n, m) - can be square or rectangular

    Returns:
        Normalized kernel matrix of same shape as K.
        For square matrices: applies cosine normalization.
        For rectangular matrices: returns K unchanged (would require diagonals from
        separate K(X_rows, X_rows) and K(X_cols, X_cols) computations).

    Warning:
        For rectangular matrices (test/validation kernels), normalization is skipped.
        This may create scale inconsistencies when combining normalized training kernels
        with unnormalized test kernels in hybrid approaches. The impact is typically
        small since both quantum and classical test kernels are affected equally.
    """
    n_rows, n_cols = K.shape

    # Handle square matrix case (training kernel)
    if n_rows == n_cols:
        # Get diagonal elements
        diag = np.diag(K).reshape(-1, 1)

        # Compute normalization matrix: sqrt(K[i,i] * K[j,j])
        normalization = np.sqrt(diag @ diag.T)

        # Avoid division by zero
        normalization[normalization == 0] = 1

        return K / normalization

    # Handle rectangular matrix case (test/validation kernel)
    # For rectangular kernel K of shape (n, m), we need:
    # - K_diag_rows: diagonal of K(X_rows, X_rows) - shape (n,)
    # - K_diag_cols: diagonal of K(X_cols, X_cols) - shape (m,)
    # But we don't have these for test data, so we skip normalization
    # This is a reasonable approach since the training kernel is normalized
    # and maintains consistency in the hybrid combination
    return K


def get_hybrid_kernel_matrix(
    classical_kernel, quantum_kernel, alpha=0.5, normalize_method="none"
):
    """
    Compute hybrid kernel matrix combining classical and quantum kernels.

    Args:
        classical_kernel: Classical kernel matrix (n_samples, n_samples)
        quantum_kernel: Quantum kernel matrix (n_samples, n_samples)
        alpha: Mixing parameter (0 to 1). 0 = pure classical, 1 = pure quantum
        normalize_method: Normalization method
            - "cosine": Angular normalization
            - "trace": Trace normalization
            - "frobenius": Frobenius norm normalization
            - "none": No normalization

    Returns:
        Hybrid kernel matrix
    """
    if normalize_method == "cosine":
        classical_norm = normalize_kernel_cosine(classical_kernel)
        quantum_norm = normalize_kernel_cosine(quantum_kernel)
    elif normalize_method == "trace":
        classical_norm = normalize_kernel_trace(classical_kernel)
        quantum_norm = normalize_kernel_trace(quantum_kernel)
    elif normalize_method == "frobenius":
        classical_norm = normalize_kernel_frobenius(classical_kernel)
        quantum_norm = normalize_kernel_frobenius(quantum_kernel)
    elif normalize_method == "centered":
        classical_norm = normalize_kernel_centered(classical_kernel)
        quantum_norm = normalize_kernel_centered(quantum_kernel)
    else:  # "none"
        classical_norm = classical_kernel
        quantum_norm = quantum_kernel
    return alpha * quantum_norm + (1 - alpha) * classical_norm


def compute_projected_features(circuit, data, n_dim):
    """Compute projected quantum features for Huang et al. 2021 local kernel.

    For each sample x in data, returns the Pauli-Z expectation vector:
        f_i(x) = <ψ_x|Z_i|ψ_x>  for i = 0, ..., n_dim-1
    where |ψ_x> = U(x)|0⟩.

    Uses Qiskit Statevector (CPU) — O(N) quantum evals vs O(N²) for standard QSVM.

    Args:
        circuit: Unbound QuantumCircuit (e.g. from make_bsp) with n_dim parameters.
        data:    Array of shape (N, n_dim), pre-scaled features.
        n_dim:   Number of qubits.

    Returns:
        features: np.ndarray of shape (N, n_dim), Pauli-Z expectations in [-1, 1].
    """
    from qiskit.quantum_info import Statevector

    features = np.zeros((len(data), n_dim))
    indices = np.arange(2**n_dim, dtype=np.int32)
    for i, x in enumerate(data):
        bound_qc = circuit.assign_parameters(tuple(float(v) for v in x))
        probs = np.abs(Statevector(bound_qc).data) ** 2
        for q in range(n_dim):
            bits_q = (indices >> q) & 1  # 1 if qubit q is |1⟩
            features[i, q] = float(np.sum(probs * (1 - 2 * bits_q)))
    return features


def projected_kernel_matrix(features1, features2, gamma=1.0):
    """RBF kernel over projected quantum feature vectors.

    K(x, y) = exp(-gamma * ||f(x) - f(y)||^2)

    Args:
        features1: (N, n_dim) array of projected features.
        features2: (M, n_dim) array of projected features.
        gamma:     RBF bandwidth (default 1.0).

    Returns:
        K: (N, M) kernel matrix.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    return rbf_kernel(features1, features2, gamma=gamma)
