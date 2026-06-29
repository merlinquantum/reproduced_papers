import merlin
import sklearn
import torch
from lib.metrics import calculate_kernel_distance_F,calculate_eta_max,calculate_g,RBF,RBF_2
from lib.ploting import plot
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.decomposition import PCA
from lib.imports import data
from datetime import datetime


class result():
    def __init__(self, g, var_FQK, var_RBF, var_RBF_order_2, F,eta_max, ROC_AUC):
        self.var_FQK = var_FQK
        self.var_RBF = var_RBF
        self.var_RBF_order_2 = var_RBF_order_2
        self.g = g
        self.F = F
        self.eta_max = eta_max
        self.ROC_AUC = ROC_AUC

def subset_PCA(X_train, y_train, X_test, y_test, nb_train, nb_test, dim = -1, seed = 42):
    """Extrait un sous-ensemble des données et applique PCA pour réduire la dimensionnalité."""
    
    torch.manual_seed(seed)  # Pour la reproductibilité
    
    indices_train = torch.randperm(X_train.size(0))[:nb_train]
    indices_test = torch.randperm(X_test.size(0))[:nb_test]
   
    X_train_subset = X_train[indices_train].view(nb_train, -1).numpy()  # Aplatir les images
    y_train_subset = y_train[indices_train]
    X_test_subset = X_test[indices_test].view(nb_test, -1).numpy()  # Aplatir les images
    y_test_subset = y_test[indices_test]
    
    if dim != -1:
        # Appliquer PCA pour réduire à 'dim' dimensions
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train_subset)
        X_test = pca.transform(X_test_subset)
    
    return torch.from_numpy(X_train).float(), y_train_subset, torch.from_numpy(X_test).float(), y_test_subset

def train(X_train, y_train_1D, X_test, y_test_1D, bandwidth = 1.0, n_modes = -1):

    X_train = X_train * bandwidth
    X_test = X_test * bandwidth

    if n_modes == -1:
        n_modes = X_train.shape[1] + 1
    if n_modes < X_train.shape[1] + 1:
        raise ValueError(f"n_modes must be at least {X_train.shape[1] + 1} for the given input size.")
    if (n_modes - 1) % X_train.shape[1] != 0:
         raise ValueError(f"the number of coding modes must be a multiple of {X_train.shape[1]}, you give {n_modes - 1}")
    
    n = int((n_modes - 1) / X_train.shape[1])

    X_train_mod = torch.cat([X_train * i for i in range(1,n+1)],axis=1)
    X_test_mod = torch.cat([X_test * i for i in range(1,n+1)],axis=1)

    builder = merlin.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, model="mzi", name="left")
    builder.add_angle_encoding(modes=[i for i in range(X_train.shape[1])], name="phi")
    builder.add_entangling_layer(trainable=True, model="mzi", name="right")
    
    feature_map = merlin.FeatureMap(builder=builder, input_size=n_modes-1, input_parameters="phi")

    fidelity_kernel = merlin.FidelityKernel(
        feature_map=feature_map,
        input_state=[1 - (i % 2) for i in range(n_modes)],  # alternating photons for n_modes
        computation_space=merlin.ComputationSpace.FOCK,
    )
    
    svc = sklearn.svm.SVC(kernel="precomputed")

    K_train = fidelity_kernel(X_train_mod)
    K_test = fidelity_kernel(X_test_mod, X_train_mod)


    svc.fit(K_train.detach().numpy(), y_train_1D.detach().numpy())


    F = calculate_kernel_distance_F(K_train, K_rbf)
    eta_max = calculate_eta_max(K_train)
    ROC_AUC = sklearn.metrics.roc_auc_score(y_test_1D.detach().numpy(), svc.decision_function(K_test.detach().numpy()))
    K_rbf = RBF(X_train_mod)
    K_rbf_order_2 = RBF_2(X_train_mod)


    return result(calculate_g(K_train,K_rbf).item(), K_train.var(correction=False).item(), K_rbf.var(correction=False).item(), K_rbf_order_2.var(correction=False).item(), F.item(), eta_max.item(), ROC_AUC)

def _run_experiment(cfg: dict[str, Any]):
    seed = int(cfg['seed'])

    now = datetime.now()
    result_folder_name = now.strftime("%Y.%m.%d-%H.%M.%S")
    print(result_folder_name)
    path = Path(__file__).parents[1]
    new_folder = Path(path / "results" / result_folder_name)
    new_folder.mkdir(parents = True)

    # Bandwidths (logarithmically spaced)
    for i in range(len(cfg['experiments'])):
        exp = cfg['experiments'][i]
        MIN,MAX,NB_Points = exp['graphs']["min"],exp['graphs']["max"],exp['graphs']["number_of_points"]

        # Stockage des résultats pour chaque métrique
        x,y_g,y_FQK,y_RBF,y_RBF_order_2,y_F,y_eta_max,y_ROC_AUC = np.logspace(MIN, MAX, NB_Points),np.zeros(NB_Points),np.zeros(NB_Points),np.zeros(NB_Points),np.zeros(NB_Points),np.zeros(NB_Points),np.zeros(NB_Points),np.zeros(NB_Points)

        # Size of the training and testing datasets
        NB_TRAIN = exp['train_sample']
        NB_TEST = exp['test_sample']

        SEEDS = np.random.default_rng(seed).integers(low=0,high=100,size=5)

    

        print(f"experiment {exp['description']} running")
        X_train, y_train, X_test, y_test = data(cfg['dataset']['name'])

        for seed in SEEDS:
            print(f"------ SEED : {seed} ------")
            X_train, y_train, X_test, y_test = subset_PCA(X_train, y_train, X_test, y_test, nb_train=NB_TRAIN, nb_test=NB_TEST, dim = 4, seed = seed)
            for i in range(NB_Points):
                res = train(X_train, y_train, X_test, y_test, bandwidth=x[i], n_modes=exp['coding_modes']+1)
                y_g[i] += res.g
                y_FQK[i] += res.var_FQK
                y_RBF[i] += res.var_RBF
                y_RBF_order_2[i] += res.var_RBF_order_2
                y_F[i] += res.F
                y_eta_max[i] += res.eta_max
                y_ROC_AUC[i] += res.ROC_AUC

        y_g_avg = y_g / len(SEEDS)
        y_FQK_avg = y_FQK / len(SEEDS)
        y_RBF_avg = y_RBF / len(SEEDS)
        y_RBF_order_2_avg = y_RBF_order_2 / len(SEEDS)
        y_F_avg = y_F / len(SEEDS)
        y_eta_max_avg = y_eta_max / len(SEEDS)
        y_ROC_AUC_avg = y_ROC_AUC / len(SEEDS)

        plot(x,y_g_avg,y_FQK_avg,y_RBF_avg,y_RBF_order_2_avg,y_F_avg,y_eta_max_avg,y_ROC_AUC_avg,new_folder,exp['figs'],exp['description'])
        print("done")