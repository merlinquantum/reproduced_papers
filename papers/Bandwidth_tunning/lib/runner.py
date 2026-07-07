import logging
from pathlib import Path
from typing import Any

import merlin
import numpy as np
import sklearn
import torch
from lib.imports import data
from lib.metrics import (
    RBF,
    RBF_2,
    calculate_eta_max,
    calculate_g,
    calculate_kernel_distance_F,
    fidelity_kernel,
    projected_fidelity_kernel,
)
from lib.ploting import overlapping_plot
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class result:
    def __init__(self, g, var_FQK, var_RBF, F, eta_max_Q, eta_max_C, ROC_AUC):
        self.var_FQK = var_FQK
        self.var_RBF = var_RBF
        self.g = g
        self.F = F
        self.eta_max_Q = eta_max_Q
        self.eta_max_C = eta_max_C
        self.ROC_AUC = ROC_AUC


def subset_PCA(X_train, y_train, X_test, y_test, nb_train, nb_test, dim=-1, seed=42):
    """Extrait un sous-ensemble des données et applique PCA pour réduire la dimensionnalité."""

    torch.manual_seed(seed)  # Pour la reproductibilité

    indices_train = torch.randperm(X_train.size(0))[:nb_train]
    indices_test = torch.randperm(X_test.size(0))[:nb_test]

    X_train_subset = (
        X_train[indices_train].view(nb_train, -1).numpy()
    )  # Aplatir les images
    y_train_subset = y_train[indices_train]
    X_test_subset = X_test[indices_test].view(nb_test, -1).numpy()  # Aplatir les images
    y_test_subset = y_test[indices_test]

    if dim != -1:
        # Appliquer PCA pour réduire à 'dim' dimensions
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train_subset)
        X_test = pca.transform(X_test_subset)

    return (
        torch.from_numpy(X_train).float(),
        y_train_subset,
        torch.from_numpy(X_test).float(),
        y_test_subset,
    )


def train(feature_map, X_train, y_train_1D, X_test, y_test_1D, bandwidth=1.0,projected=False):
    X_train = X_train * bandwidth
    X_test = X_test * bandwidth

    #Kernel calculation
    if not projected:
        K_train = fidelity_kernel(feature_map, X_train)
        K_test = fidelity_kernel(feature_map, X_train, X_test)
        K_rbf = RBF(X_train)
    else:
        K_train = projected_fidelity_kernel(feature_map, X_train)
        K_test = projected_fidelity_kernel(feature_map, X_train, X_test)
        K_rbf = RBF_2(X_train)

    svc = sklearn.svm.SVC(kernel="precomputed")
    svc.fit(K_train.detach().numpy(), y_train_1D.detach().numpy())

    g = calculate_g(K_rbf, K_train)

    F = calculate_kernel_distance_F(K_rbf, K_train)

    eta_max_Q = calculate_eta_max(K_train)
    eta_max_C = calculate_eta_max(K_rbf)

    ROC_AUC = sklearn.metrics.roc_auc_score(
        y_test_1D.detach().numpy(), svc.decision_function(K_test.detach().numpy())
    )

    return result(
        g.item(),
        K_train.var(correction=False).item(),
        K_rbf.var(correction=False).item(),
        F.item(),
        eta_max_Q.item(),
        eta_max_C.item(),
        ROC_AUC,
    )

def run_overlapping(cfg, new_folder):
    # importing the parameters of the experiments
    seed = int(cfg["seed"])
    scale = cfg["graphs"]
    nb_of_experiments = len(cfg["experiments"])

    y_g_list = []
    y_FQK_list = []
    y_RBF_list = []
    y_F_list = []
    y_eta_max_Q_list = []
    y_eta_max_C_list = []
    y_ROC_AUC_list = []

    MIN, MAX, NB_Points = (
            scale["min"],
            scale["max"],
            scale["number_of_points"],
    )

    for i in range(nb_of_experiments):
        # running the experiments one by one
        exp = cfg["experiments"][i]


        # Stockage des résultats pour chaque métrique
        x, y_g, y_FQK, y_RBF, y_F, y_eta_max_Q, y_eta_max_C, y_ROC_AUC = (
            np.logspace(MIN, MAX, NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
            np.zeros(NB_Points),
        )

        # Size of the training and testing datasets
        NB_TRAIN = exp["train_sample"]
        NB_TEST = exp["test_sample"]

        SEEDS = np.random.default_rng(seed).integers(low=0, high=100, size=exp["nb_seeds"])

        print(f"experiment {exp['description']} running")
        X_train, y_train, X_test, y_test = data(cfg["dataset"]["name"])

        for seed in range(exp["nb_seeds"]):
            X_train, y_train, X_test, y_test = subset_PCA(
                X_train,
                y_train,
                X_test,
                y_test,
                nb_train=NB_TRAIN,
                nb_test=NB_TEST,
                dim=exp["dimension"],
                seed=SEEDS[seed],
            )

            #Circuit Building (once for every seed, since the feature map is fixed for a given dataset)
            builder = merlin.CircuitBuilder(n_modes=X_train.shape[1] + 1)
            builder.add_entangling_layer(trainable=True, model="mzi", name="left")
            builder.add_angle_encoding(modes=range(X_train.shape[1]), name="phi")
            builder.add_entangling_layer(trainable=True, model="mzi", name="right")

            feature_map = merlin.FeatureMap(
                builder=builder, input_size=X_train.shape[1], input_parameters="phi"
            )

            for i in range(NB_Points):
                res = train(feature_map, X_train, y_train, X_test, y_test, bandwidth=x[i], projected=exp["projected"])
                y_g[i] += res.g
                y_FQK[i] += res.var_FQK
                y_RBF[i] += res.var_RBF
                y_F[i] += res.F
                y_eta_max_Q[i] += res.eta_max_Q
                y_eta_max_C[i] += res.eta_max_C
                y_ROC_AUC[i] += res.ROC_AUC
                print(
                    f"experiment {exp['description']} running, seed {SEEDS[seed]}, bandwidth {x[i]} done ({(i+1+NB_Points*seed)/(NB_Points*len(SEEDS))*100:.2f}%)"
                )

        # averaging the results over the different seeds
        y_g_avg = y_g / len(SEEDS)
        y_FQK_avg = y_FQK / len(SEEDS)
        y_RBF_avg = y_RBF / len(SEEDS)
        y_F_avg = y_F / len(SEEDS)
        y_eta_max_Q_avg = y_eta_max_Q / len(SEEDS)
        y_eta_max_C_avg = y_eta_max_C / len(SEEDS)
        y_ROC_AUC_avg = y_ROC_AUC / len(SEEDS)

        # Storing the results for overlapping plots
        y_g_list.append(y_g_avg.copy())
        y_FQK_list.append(y_FQK_avg.copy())
        y_RBF_list.append(y_RBF_avg.copy())
        y_F_list.append(y_F_avg.copy())
        y_eta_max_Q_list.append(y_eta_max_Q_avg.copy())
        y_eta_max_C_list.append(y_eta_max_C_avg.copy())
        y_ROC_AUC_list.append(y_ROC_AUC_avg.copy())

        legends = [exp["dimension"] for exp in cfg["experiments"]]

    # Calling the overlapping_plot function to generate the plots
    overlapping_plot(
        x,
        y_g_list,
        y_FQK_list,
        y_RBF_list,
        y_F_list,
        y_eta_max_Q_list,
        y_eta_max_C_list,
        y_ROC_AUC_list,
        new_folder,
        cfg["figs"],
        legends,
        cfg["graph_name"],
        exp["projected"],
    )


def _run_experiment(cfg: dict[str, Any], run_dir: Path):
    run_overlapping(cfg, run_dir)



def train_and_evaluate(cfg: dict[str, Any], run_dir):
    run_dir = Path(run_dir)
    _run_experiment(cfg, run_dir)
    (run_dir / "done.txt").write_text("Completed")
    logger.info("Finished. Artifacts in: %s", run_dir)


__all__ = ["train_and_evaluate"]
