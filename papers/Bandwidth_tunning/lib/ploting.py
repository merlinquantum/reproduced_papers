import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
            ax.loglog(x, y_FQK_avg, label="FQK", linestyle='-')
            ax.loglog(x, y_RBF_avg, label="RBF", linestyle='--')
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

import matplotlib.pyplot as plt

def overlapping_plot(x, y_g_avg, y_FQK_avg, y_RBF_avg, y_RBF_order_2_avg, y_F_avg, y_eta_max_avg, y_ROC_AUC_avg, folder_name, list_of_plots,legendes, exp_name):
    
    n_plots = len(list_of_plots)
    
    # Si la liste est vide, on arrête l'exécution
    if n_plots == 0:
        raise ValueError("There must be at least 1 plot to generate")

    # 1. Création de la figure sur 1 seule ligne
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # 2. Gestion de l'exception Matplotlib si n_plots == 1
    if n_plots == 1:
        axes = [axes]

    # Palette de couleurs personnalisée calquée sur ta capture d'écran 
    # (du beige/pêche clair au violet/noir très foncé)
    couleurs = ['#f0b593', '#d65f49', '#662248', '#2b1b36']
    
    # Épaisseur des traits pour correspondre à l'esthétique du screenshot
    lw = 3

    # 3. Boucle principale sur les graphiques demandés
    for i, plot_name in enumerate(list_of_plots):
        ax = axes[i] # Sélectionne le sous-graphique
        
        # 4. Boucle secondaire : on trace les courbes pour chaque cas
        for j, legende in enumerate(legendes):
            c = couleurs[j % len(couleurs)] # Couleur unique par cas (boucle si plus de 4 éléments)
            
            if plot_name == "Variances":
                ax.loglog(x, y_FQK_avg[j], label=f"FQK ({legende})", color=c, linestyle='-', linewidth=lw)
                ax.loglog(x, y_RBF_avg[j], label=f"RBF ({legende})", color=c, linestyle='--', linewidth=lw)

            elif plot_name == "Geometric_distance":
                # Ligne dash-dot (-.) pour correspondre au graphe (D1) de l'image
                ax.loglog(x, y_g_avg[j], label=legende, color=c, linestyle='-.', linewidth=lw)

            elif plot_name == "Frobenius_distance":
                # Ligne dash-dot (-.) pour correspondre au graphe (E1) de l'image
                ax.loglog(x, y_F_avg[j], label=legende, color=c, linestyle='-.', linewidth=lw)

            elif plot_name == "Eta_max":
                # Ligne continue comme le graphe (B1) de l'image
                ax.loglog(x, y_eta_max_avg[j], label=legende, color=c, linestyle='-', linewidth=lw)

            elif plot_name == "ROC_AUC":
                ax.semilogx(x, y_ROC_AUC_avg[j], label=legende, color=c, linestyle='-', linewidth=lw)

            elif plot_name == "VAR_RBF_2":
                ax.loglog(x, y_RBF_order_2_avg[j], label=legende, color=c, linestyle='-', linewidth=lw)
                
            else:
                raise NameError(f"'{plot_name}' is not a valid name of plot")

        # 5. Configuration du sous-graphique
        if plot_name == "Variances":
            ax.set_title("Variances des noyaux")
            ax.set_ylabel(r"$Var_D[\mathbf{K}]$")
        elif plot_name == "Geometric_distance":
            ax.set_ylabel(r"$g(\mathbf{K}_C, \mathbf{K}_Q)$")
        elif plot_name == "Frobenius_distance":
            ax.set_ylabel(r"$F(\mathbf{K}_C, \mathbf{K}_Q)$")
        elif plot_name == "Eta_max":
            ax.set_ylabel(r"$\eta_{max(K)}$")
        elif plot_name == "ROC_AUC":
            ax.set_ylabel("roc auc score")
        elif plot_name == "VAR_RBF_2":
            ax.set_ylabel("Variance of RBF order 2")

        # Ces éléments sont communs à tous les graphiques
        ax.set_xlabel(r"Bandwidth $c$")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    # Affichage propre et Sauvegarde
    # ==========================================
    fig.suptitle(exp_name)
    plt.tight_layout()
    
    results_folder = folder_name / f"{exp_name}.png"
    plt.savefig(results_folder)
    
    plt.close()
