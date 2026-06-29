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

def overlapping_plot(x, y_g_avg, y_FQK_avg, y_RBF_avg, y_RBF_order_2_avg, y_F_avg, y_eta_max_avg, y_ROC_AUC_avg, folder_name, list_of_plots, legends, title):
    
    n_plots = len(list_of_plots)
    
    # Si la liste est vide, on arrête l'exécution
    if n_plots == 0:
        raise ValueError("There must be at least 1 plot to generate")

    # 1. Création de la figure sur 1 seule ligne
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # 2. Gestion de l'exception Matplotlib si n_plots == 1
    if n_plots == 1:
        axes = [axes]

    # Palette de couleurs pour différencier les cas (les éléments de "legendes")
    couleurs = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 3. Boucle principale sur les graphiques demandés
    for i, plot_name in enumerate(list_of_plots):
        ax = axes[i] # Sélectionne le sous-graphique
        
        # 4. Boucle secondaire : on trace les courbes pour chaque cas
        for j, legend in enumerate(legends):
            c = couleurs[j % len(couleurs)] # Couleur unique par cas
            
            if plot_name == "Variances":
                # Lignes continues/pointillées, SANS marqueurs (comme demandé précédemment)
                ax.loglog(x, y_FQK_avg[j], label=f"FQK ({legend})", color=c, linestyle='-')
                ax.loglog(x, y_RBF_avg[j], label=f"RBF ({legend})", color=c, linestyle='--')

            elif plot_name == "Geometric_distance":
                ax.loglog(x, y_g_avg[j], label=legend, color=c, marker='d', linestyle='-')

            elif plot_name == "Frobenius_distance":
                ax.loglog(x, y_F_avg[j], label=legend, color=c, marker='v', linestyle='-')

            elif plot_name == "Eta_max":
                ax.loglog(x, y_eta_max_avg[j], label=legend, color=c, marker='^', linestyle='-')

            elif plot_name == "ROC_AUC":
                ax.semilogx(x, y_ROC_AUC_avg[j], label=legend, color=c, marker='s', linestyle='-')

            elif plot_name == "VAR_RBF_2":
                ax.loglog(x, y_RBF_order_2_avg[j], label=legend, color=c, marker='x', linestyle='-')
                
            else:
                raise NameError(f"'{plot_name}' is not a valid name of plot")

        # 5. Configuration du sous-graphique (faite UNE SEULE FOIS par graphique, hors de la boucle des cas)
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
    # On utilise title pour le titre général de la figure
    fig.suptitle(title)
    plt.tight_layout()
    
    # On sauvegarde avec le nom de l'expérience pour retrouver facilement le fichier
    results_folder = folder_name / f"{title}.png"
    plt.savefig(results_folder)
    
    plt.close()