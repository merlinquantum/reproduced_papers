import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot0(x,y_g_avg,y_FQK_avg,y_RBF_avg,y_RBF_order_2_avg,y_F_avg,y_eta_max_avg,y_ROC_AUC_avg, folder_name):
    # Création de la figure et d'une grille de 6 graphiques
    # figsize=(15, 5) permet d'avoir une image bien large
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ==========================================
    # Subplot 1 : Les Variances (FQK et RBF)
    # ==========================================
    axes[0,0].loglog(x, y_FQK_avg, label="FQK", marker='o', linestyle='-')
    axes[0,0].loglog(x, y_RBF_avg, label="RBF", marker='s', linestyle='-')

    axes[0,0].set_title("Variances des noyaux")
    axes[0,0].set_xlabel(r"Bandwidth $c$")
    axes[0,0].set_ylabel(r"$Var_D[\mathbf{K}]$")
    axes[0,0].legend() # Affiche la légende pour différencier les 3 courbes
    axes[0,0].grid(True, which="both", ls="--", alpha=0.5) # Ajoute une grille lisible en log

    # ==========================================
    # Subplot 2 : La métrique g
    # ==========================================
    axes[0,1].loglog(x, y_g_avg, color='purple', marker='d', linestyle='-')

    axes[0,1].set_title(r"Métrique $g$")
    axes[0,1].set_xlabel(r"Bandwidth $c$")
    axes[0,1].set_ylabel(r"$g(\mathbf{K}_C, \mathbf{K}_Q)$")
    axes[0,1].grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # Subplot 3 : La distance F
    # ==========================================
    axes[1,0].loglog(x, y_F_avg, color='green', marker='v', linestyle='-')

    axes[1,0].set_title("Distance de Frobenius")
    axes[1,0].set_xlabel(r"Bandwidth $c$")
    axes[1,0].set_ylabel(r"$F(\mathbf{K}_C, \mathbf{K}_Q)$")
    axes[1,0].grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # Subplot 4 : eta_max
    # ==========================================
    axes[1,1].loglog(x, y_eta_max_avg, color='orange', marker='^', linestyle='-')

    axes[1,1].set_title(r"$\eta_{max}$")
    axes[1,1].set_xlabel(r"Bandwidth $c$")
    axes[1,1].set_ylabel(r"$\eta_{max(K)}$")
    axes[1,1].grid(True, which="both", ls="--", alpha=0.5)


    # ==========================================
    # Subplot 5 : ROC AUC
    # ==========================================
    axes[0,2].semilogx(x, y_ROC_AUC_avg, color='red', marker='s', linestyle='-')

    axes[0,2].set_title(r"ROC AUC score")
    axes[0,2].set_xlabel(r"Bandwidth $c$")
    axes[0,2].set_ylabel("roc auc score")
    axes[0,2].grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # Subplot 6 : VAR(RBF_order_2)
    # ==========================================
    axes[1,2].loglog(x, y_RBF_order_2_avg, color='blue', marker='x', linestyle='-')

    axes[1,2].set_title("Variance du noyau RBF (ordre 2)")
    axes[1,2].set_xlabel(r"Bandwidth $c$")
    axes[1,2].set_ylabel("Variance")
    axes[1,2].grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # Affichage propre
    # ==========================================
    fig.suptitle(r"kMNIST28 $(N = , PCA = 4, Seeds = 5)$")
    # tight_layout empêche les titres et les labels de se chevaucher
    plt.tight_layout()
    parent_folder = Path(__file__).parents[1]
    results_folder = folder_name / "mon_graphique"
    plt.savefig(results_folder)


def plot(x, y_g_avg, y_FQK_avg, y_RBF_avg, y_RBF_order_2_avg, y_F_avg, y_eta_max_avg, y_ROC_AUC_avg, folder_name, list_of_plots, exp_name):
        
    n_plots = len(list_of_plots)
    
    # Si la liste est vide, on arrête l'exécution de la fonction ici
    if n_plots == 0:
        raise ValueError("There must be at least 1 plot to generate")

    # 1. Création de la figure sur 1 seule ligne
    # La largeur s'adapte dynamiquement (5 pouces par graphique)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # 2. Gestion de l'exception Matplotlib si n_plots == 1
    # Si on a qu'un seul plot, "axes" n'est pas une liste. On la force en liste.
    if n_plots == 1:
        axes = [axes]

    # 3. Boucle sur les graphiques demandés
    for i, plot_name in enumerate(list_of_plots):
        ax = axes[i] # On sélectionne l'emplacement i sur la ligne
        
        if plot_name == "Variances":
            ax.loglog(x, y_FQK_avg, label="FQK", marker='o', linestyle='-')
            ax.loglog(x, y_RBF_avg, label="RBF", marker='s', linestyle='-')
            ax.set_title("Variances des noyaux")
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel(r"$Var_D[\mathbf{K}]$")
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)

        elif plot_name == "Geometric_distance":
            ax.loglog(x, y_g_avg, color='purple', marker='d', linestyle='-')
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel(r"$g(\mathbf{K}_C, \mathbf{K}_Q)$")
            ax.grid(True, which="both", ls="--", alpha=0.5)

        elif plot_name == "Frobenius_distance":
            ax.loglog(x, y_F_avg, color='green', marker='v', linestyle='-')
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel(r"$F(\mathbf{K}_C, \mathbf{K}_Q)$")
            ax.grid(True, which="both", ls="--", alpha=0.5)

        elif plot_name == "Eta_max":
            ax.loglog(x, y_eta_max_avg, color='orange', marker='^', linestyle='-')
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel(r"$\eta_{max(K)}$")
            ax.grid(True, which="both", ls="--", alpha=0.5)

        elif plot_name == "ROC_AUC":
            ax.semilogx(x, y_ROC_AUC_avg, color='red', marker='s', linestyle='-')
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel("roc auc score")
            ax.grid(True, which="both", ls="--", alpha=0.5)

        elif plot_name == "VAR_RBF_2":
            ax.loglog(x, y_RBF_order_2_avg, color='blue', marker='x', linestyle='-')
            ax.set_xlabel(r"Bandwidth $c$")
            ax.set_ylabel("Variance of RBF order 2")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            
        else:
            raise NameError(f"{plot_name} is not a valid name of plot")

    # ==========================================
    # Affichage propre et Sauvegarde
    # ==========================================
    fig.suptitle(exp_name)
    plt.tight_layout()
    
    figure_name = exp_name + ".png"
    # Construction du chemin cible et ajout de l'extension .png
    results_folder = folder_name / figure_name
    plt.savefig(results_folder)
    
    # Toujours clore la figure pour libérer la mémoire en boucle
    plt.close()