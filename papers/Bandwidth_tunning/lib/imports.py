import gzip
import torch
import numpy as np
import os
from pathlib import Path


#==============================================================================
#
#                              fashion_kmnist
#
#==============================================================================

def charger_images_fashion_mnist_torch(chemin_fichier):
    """Lit le fichier d'images et retourne un tenseur PyTorch normalisé (B, C, H, W)."""
    with gzip.open(chemin_fichier, 'rb') as f:
        donnees = np.frombuffer(f.read(), np.uint8, offset=16)
    data = np.asarray(donnees, dtype=np.float32).copy()
    # 1. Conversion en tableau NumPy puis en Tenseur PyTorch
    images_np = data.reshape(-1, 28, 28)
    un_tensor = torch.from_numpy(images_np)
   
    # 2. Conversion en float32 et normalisation entre 0.0 et 1.0
    un_tensor = un_tensor.float() / 255.0
   
    # 3. Ajout de la dimension du canal (grayscale -> 1 canal) à la position 1
    # Passe de (60000, 28, 28) à (60000, 1, 28, 28)
    un_tensor = un_tensor.unsqueeze(1)
   
    return un_tensor

def charger_labels_fashion_mnist_torch(chemin_fichier):
    """Lit le fichier de labels et retourne un tenseur PyTorch de type Long."""
    with gzip.open(chemin_fichier, 'rb') as f:
        donnees = np.frombuffer(f.read(), np.uint8, offset=8)
   
    # Conversion en tenseur et forçage en type Long (requis pour la classification)
    return torch.from_numpy(donnees).long()

def load_fashion_mnist_torch():
    """Charge les données Fashion-MNIST et retourne les tenseurs PyTorch pour l'entraînement et le test."""
    path = Path(__file__).resolve()
    root = path.parents[3]

    dossier_base = root / "data" / "Bandwidth_tunning" / "fashion_mnist"

    chemin_train_images = dossier_base / 'train-images-idx3-ubyte.gz'
    chemin_train_labels = dossier_base / 'train-labels-idx1-ubyte.gz'
    chemin_test_images = dossier_base / 't10k-images-idx3-ubyte.gz'
    chemin_test_labels = dossier_base / 't10k-labels-idx1-ubyte.gz'

    X_train_tensor = charger_images_fashion_mnist_torch(chemin_train_images)
    y_train_tensor = charger_labels_fashion_mnist_torch(chemin_train_labels)

    X_test_tensor = charger_images_fashion_mnist_torch(chemin_test_images)
    y_test_tensor = charger_labels_fashion_mnist_torch(chemin_test_labels)

    masque_train = (y_train_tensor == 2) | (y_train_tensor == 8)
    masque_test = (y_test_tensor == 2) | (y_test_tensor == 8)

    X_train_tensor = X_train_tensor[masque_train]
    y_train_tensor = y_train_tensor[masque_train]

    X_test_tensor = X_test_tensor[masque_test]
    y_test_tensor = y_test_tensor[masque_test]

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

#==============================================================================
#
#                                 kmnist
#
#==============================================================================

def charger_npz_torch(chemin_fichier, est_image=True):
    """
    Charge un fichier .npz et le convertit en tenseur PyTorch.
    """
    # 1. Chargement de l'archive NumPy
    archive = np.load(chemin_fichier)
   
    # Les fichiers .npz sont comme des dictionnaires. On récupère le premier tableau.
    nom_tableau = archive.files[0]
    donnees_np = archive[nom_tableau]
   
    # 2. Conversion en tenseur PyTorch
    tenseur = torch.from_numpy(donnees_np)
   
    # 3. Formatage spécifique selon que ce soit une image ou un label
    if est_image:
        # Conversion en float32, normalisation (0 à 1) et ajout du canal (B, 1, H, W)
        tenseur = tenseur.float() / 255.0
        tenseur = tenseur.unsqueeze(1)
    else:
        # Forçage en type Long pour les labels
        tenseur = tenseur.long()
       
    return tenseur

def load_kmnist28():
    path = Path(__file__).resolve()
    root = path.parents[3]

    dossier_kmnist = root / "data" / "Bandwidth_tunning" / "kmnist"

    # Noms exacts d'après votre capture d'écran
    chemin_train_imgs = dossier_kmnist / 'kmnist-train-imgs.npz'
    chemin_train_labels = dossier_kmnist / 'kmnist-train-labels.npz'
    chemin_test_imgs = dossier_kmnist / 'kmnist-test-imgs.npz'
    chemin_test_labels = dossier_kmnist / 'kmnist-test-labels.npz'

    # --- CHARGEMENT DES DONNÉES ---
    X_train = charger_npz_torch(chemin_train_imgs, est_image=True)
    y_train = charger_npz_torch(chemin_train_labels, est_image=False)

    X_test = charger_npz_torch(chemin_test_imgs, est_image=True)
    y_test = charger_npz_torch(chemin_test_labels, est_image=False)

    masque_train = (y_train == 2) | (y_train == 8)
    masque_test = (y_test == 2) | (y_test == 8)

    X_train = X_train[masque_train]
    y_train = y_train[masque_train]

    X_test = X_test[masque_test]
    y_test = y_test[masque_test]

    return X_train, y_train, X_test, y_test


#==============================================================================
#
#                            hidden_manifold
#
#==============================================================================


def load_hidden_manifold():
    from hidden_manifold import generate_hidden_manifold_model
    X,y = generate_hidden_manifold_model(400,10)
    return X[:320],y[:320],X[320:],y[320:]


#==============================================================================
#
#                                plasticc
#
#==============================================================================

def load_plasticc():
    path = Path(__file__).resolve()
    root = path.parents[3]

    dossier_plasticc = root / "data" / "Bandwidth_tunning" / "plasticc_data"
    # Noms exacts d'après votre capture d'écran
    chemin_fichier_npy = os.path.join(dossier_plasticc, 'SN_67floats_preprocessed.npy')

    # --- CHARGEMENT DES DONNÉES ---
    donnes = np.load(chemin_fichier_npy)
    tenseur_plasticc = torch.from_numpy(donnes).float()
    X_plasticc = tenseur_plasticc[:, :-1]  # Toutes les colonnes sauf la dernière
    y_plasticc = tenseur_plasticc[:, -1].long()  # La dernière colonne comme labels

    X_train, X_test = torch.split(X_plasticc, [2500,1006],dim=0)
    y_train, y_test = torch.split(y_plasticc, [2500,1006],dim=0)

    return X_train, y_train, X_test, y_test


#==============================================================================
#
#                                 global
#
#==============================================================================


def data(dataset):
    if dataset == "fashion_mnist":
        return load_fashion_mnist_torch()
    if dataset == "kmnist28":
        return load_kmnist28()
    if dataset == "hidden_manifold":
        return load_hidden_manifold
    if dataset == "plasticc":
        return load_plasticc()
    raise ValueError(f"Unsupported dataset: {dataset}")