import matplotlib.pyplot as plt


def overlapping_plot(
    x,
    y_g_avg,
    y_FQK_avg,
    y_RBF_avg,
    y_F_avg,
    y_eta_max_Q_avg,
    y_eta_max_C_avg,
    y_ROC_AUC_avg,
    folder_name,
    list_of_plots,
    legendes,
    exp_name,
    projected,
):
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
    couleurs = ["#f0b593", "#d65f49", "#a33f5f", "#662248", "#4a2a5f", "#2b1b36"]

    # Épaisseur des traits pour correspondre à l'esthétique du screenshot
    lw = 3

    # 3. Boucle principale sur les graphiques demandés
    for i, plot_name in enumerate(list_of_plots):
        ax = axes[i]  # Sélectionne le sous-graphique

        # 4. Boucle secondaire : on trace les courbes pour chaque cas
        for j, legende in enumerate(legendes):
            c = couleurs[
                j % len(couleurs)
            ]  # Couleur unique par cas (boucle si plus de 6 éléments)

            if plot_name == "Variances":
                if isinstance(projected, (list, tuple)):
                    is_projected = bool(projected[j])
                else:
                    is_projected = bool(projected)

                qk_label = "PQK" if is_projected else "FQK"
                rbf_label = "RBF_2" if is_projected else "RBF"

                ax.loglog(
                    x,
                    y_FQK_avg[j],
                    label=f"{qk_label} ({legende})",
                    color=c,
                    linestyle="-",
                    linewidth=lw,
                )
                ax.loglog(
                    x,
                    y_RBF_avg[j],
                    label=f"{rbf_label} ({legende})",
                    color=c,
                    linestyle="--",
                    linewidth=lw,
                )

            elif plot_name == "Geometric_distance":
                # Ligne dash-dot (-.) pour correspondre au graphe (D1) de l'image
                ax.loglog(
                    x, y_g_avg[j], label=legende, color=c, linestyle="-", linewidth=lw
                )

            elif plot_name == "Frobenius_distance":
                # Ligne dash-dot (-.) pour correspondre au graphe (E1) de l'image
                ax.loglog(
                    x, y_F_avg[j], label=legende, color=c, linestyle="-", linewidth=lw
                )

            elif plot_name == "Eta_max":
                if isinstance(projected, (list, tuple)):
                    is_projected = bool(projected[j])
                else:
                    is_projected = bool(projected)

                qk_label = "PQK" if is_projected else "FQK"
                rbf_label = "RBF_2" if is_projected else "RBF"

                ax.loglog(
                    x,
                    y_eta_max_Q_avg[j],
                    label=f"eta_max_{qk_label} ({legende})",
                    color=c,
                    linestyle="-",
                    linewidth=lw,
                )
                ax.loglog(
                    x,
                    y_eta_max_C_avg[j],
                    label=f"eta_max_{rbf_label} ({legende})",
                    color=c,
                    linestyle="--",
                    linewidth=lw,
                )

            elif plot_name == "ROC_AUC":
                ax.semilogx(
                    x,
                    y_ROC_AUC_avg[j],
                    label=legende,
                    color=c,
                    linestyle="-",
                    linewidth=lw,
                )

            else:
                raise NameError(f"'{plot_name}' is not a valid name of plot")

        # 5. Configuration du sous-graphique
        if plot_name == "Variances":
            ax.set_title("Variances des noyaux")
            ax.set_ylabel(r"$Var_D[\mathbf{K}]$")
            ax.set_ylim(bottom=1e-10, top=1e4)
        elif plot_name == "Geometric_distance":
            ax.set_ylabel(r"$g(\mathbf{K}_C, \mathbf{K}_Q)$")
            ax.set_ylim(bottom=5e-1, top=1e3)
        elif plot_name == "Frobenius_distance":
            ax.set_ylabel(r"$F(\mathbf{K}_C, \mathbf{K}_Q)$")
            ax.set_ylim(bottom=1e-3, top=1e3)
        elif plot_name == "Eta_max":
            ax.set_ylabel(r"$\eta_{max(K)}$")
            ax.set_ylim(bottom=1e-3, top=1e5)
        elif plot_name == "ROC_AUC":
            ax.set_ylabel("roc auc score")

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
