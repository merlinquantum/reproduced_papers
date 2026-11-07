import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _transform_pca(pca, sample_train, sample_test):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        transformed_train = pca.transform(sample_train)
        transformed_test = pca.transform(sample_test)
    if (
        not np.isfinite(transformed_train).all()
        or not np.isfinite(transformed_test).all()
    ):
        raise ValueError("PCA produced non-finite values")
    return transformed_train, transformed_test


# EDITED: pre-computed data
def data_prepare_cv(
    n_dim,
    sample_train,
    sample_test,
    fix_leakage=False,
    pi_angles=False,
    svd_solver="auto",
):
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver=svd_solver).fit(sample_train)
    sample_train, sample_test = _transform_pca(pca, sample_train, sample_test)
    angle_range = np.pi if pi_angles else 1
    if fix_leakage:
        minmax_scale = MinMaxScaler(feature_range=(-angle_range, angle_range)).fit(
            sample_train
        )
    else:
        samples = np.append(sample_train, sample_test, axis=0)
        minmax_scale = MinMaxScaler(feature_range=(-angle_range, angle_range)).fit(
            samples
        )
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    return sample_train, sample_test
