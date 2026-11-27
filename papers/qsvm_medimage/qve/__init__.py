# Re-export useful public API
from .core import (
    build_qsvm_qc,
    compute_cpu_kernel_entries,
    compute_projected_features,
    compute_zz_kernel_entries,
    data_partition,
    data_to_operand,
    data_to_operand_3dof,
    data_to_operand_reps,
    get_from_d1,
    get_from_d2,
    get_hybrid_kernel_matrix,
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
    projected_kernel_matrix,
    renew_operand,
    renew_operand_3dof,
    renew_operand_reps,
    sin_cos,
)
from .metrics import (
    get_metrics_multiclass_case,
    get_metrics_multiclass_case_cv,
    get_metrics_multiclass_case_test,
)
from .process import data_prepare_cv
from .utils import set_seed
