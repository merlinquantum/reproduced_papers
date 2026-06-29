import torch

def matrix_sqrt(A): 
    """ Calcule la racine carrée matricielle d'une matrice symétrique semi-définie positive. """ 
    # Décomposition en valeurs propres (A = V * L * V^T) 
    L, V = torch.linalg.eigh(A)
    # Parfois, les erreurs de précision numérique créent des valeurs propres 
    # très légèrement négatives (ex: -1e-15). On les force à être positives ou nulles. 
    L = torch.clamp(L, min=0.0) 
    # Reconstruction : V * sqrt(L) * V^T 
    return V @ torch.diag(torch.sqrt(L)) @ V.T

def calculate_g(K1, K2): 
    """ Calcule g(K1, K2) avec lambda = 0. """ 
    # 1. Calcul de l'inverse de K2
    # Utilisation de pseudo-inverse (pinv) est plus sûr que inv()
    # car une matrice de noyau (K2) peut avoir un déterminant proche de 0. 
    K2_inv = torch.linalg.pinv(K2)
    # 2. Calcul de la racine carrée matricielle de K1 
    sqrt_K1 = matrix_sqrt(K1) 
    # 3. Produit matriciel central : sqrt(K1) @ K2^-1 @ sqrt(K1) 
    inner_matrix = sqrt_K1 @ K2_inv @ sqrt_K1 
    # 4. Norme spectrale (la plus grande valeur singulière, équivalente à ord=2) 
    spectral_norm = torch.linalg.matrix_norm(inner_matrix, ord=2) 
    # 5. Racine carrée finale de la formule 
    g = torch.sqrt(spectral_norm) 
    return g

def calculate_eta_max(K):
    L,V = torch.linalg.eigh(K)
    eta_max = L[-1]
    return eta_max

def calculate_kernel_distance_F(K_C, K_Q):
   """
   Calcule F(K_C, K_Q), la distance relative de Frobenius entre deux matrices de noyau.
   K_C : Tenseur PyTorch représentant la matrice de noyau classique.
   K_Q : Tenseur PyTorch représentant la matrice de noyau quantique.
   """
   # 1. Calcul du numérateur : Norme de Frobenius de la différence
   numerateur = torch.linalg.matrix_norm(K_C - K_Q, ord='fro')

   # 2. Calcul du dénominateur : Norme de Frobenius de K_Q
   denominateur = torch.linalg.matrix_norm(K_Q, ord='fro')

   # 3. Ratio final
   F = numerateur / denominateur

   return F

def RBF(X_train):
    #Calcul du kernel RBF
    distances = torch.cdist(X_train, X_train, p=2)
    distances_carré = distances ** 2
    K_rbf = torch.exp(-distances_carré)
    return K_rbf

def RBF_2(X_train):
    #Calcul du kernel RBF ordre 2
    distances = torch.cdist(X_train, X_train, p=2)
    z = distances ** 2
    K_rbf_order_2 = 1.0 - z + 0.5*(z**2)
    return K_rbf_order_2