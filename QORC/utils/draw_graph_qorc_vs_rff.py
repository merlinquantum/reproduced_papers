#!/usr/bin/env python3

# $ micromamba activate qml-cpu
# $ python utils/draw_graph_qorc_vs_rff.py

##########################################################
# Librairies loading and functions definitions

import os
import matplotlib.pyplot as plt
import pandas as pd


def aggregate_results_csv_files(outdir, f_out_aggregated_csv):
    dataframes = []
    for root, _dirs, files in os.walk(outdir):
        for file in files:
            if "f_out_results_training_rff" in file:
                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath)
                dataframes.append(df)
    if dataframes:
        df_result = pd.concat(dataframes, ignore_index=True)
        df_result = df_result.drop_duplicates()
        df_result = df_result.sort_values(by=df_result.columns.tolist())
        if not os.path.exists(f_out_aggregated_csv):
            df_result.to_csv(f_out_aggregated_csv, index=False)
            print("Saved aggregated results to csv:", f_out_aggregated_csv)
        else:
            print("Warning: File exists.")


def draw_graph_qorc_vs_rff(
    f_in_qorc_aggregated_results_csv,
    f_in_rff_aggregated_results_csv,
    features_scale,
    figsize_list,
    f_out_img,
):
    df_qorc = pd.read_csv(f_in_qorc_aggregated_results_csv)
    df_rff = pd.read_csv(f_in_rff_aggregated_results_csv)

    df_qorc = df_qorc.rename(columns={"qorc_output_size": "n_features"})
    df_rff = df_rff.rename(columns={"n_rff_features": "n_features"})

    # Filtrer df1 pour ne garder que les lignes avec n_photons == 3
    df_qorc_filt = df_qorc[df_qorc["n_photons"] == 3]

    # Trouver les valeurs communes de n_features entre les deux dataframes
    common_features = set(df_qorc_filt["n_features"]).intersection(
        set(df_rff["n_features"])
    )

    # Filtrer les deux dataframes pour ne garder que les valeurs communes
    df_qorc_common = df_qorc_filt[df_qorc_filt["n_features"].isin(common_features)]
    df_rff_common = df_rff[df_rff["n_features"].isin(common_features)]

    # Trier par n_features pour un tracé propre
    # df_qorc_common = df_qorc_common.sort_values('n_features')
    # df_rff_common  = df_rff_common.sort_values('n_features')

    # Calculer moyenne et écart-type pour QORC
    grouped_qorc = (
        df_qorc_common.groupby("n_features")
        .agg(
            mean_train_acc=("train_acc", "mean"),
            std_train_acc=("train_acc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
        )
        .reset_index()
    )

    # Calculer moyenne et écart-type pour RFF
    grouped_rff = (
        df_rff_common.groupby("n_features")
        .agg(
            mean_train_acc=("train_acc", "mean"),
            std_train_acc=("train_acc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
        )
        .reset_index()
    )

    # Tracer les courbes
    figsize = (figsize_list[0], figsize_list[1])
    plt.figure(figsize=figsize)

    # plt.plot(df_qorc_common['n_features'], df_qorc_common['train_acc'], label='Train Acc qorc', marker='o')
    # plt.plot(df_qorc_common['n_features'], df_qorc_common['test_acc'], label='Test Acc qorc', marker='o')
    # plt.plot(df_rff_common['n_features'], df_rff_common['train_acc'], label='Train Acc RFF', marker='s')
    # plt.plot(df_rff_common['n_features'], df_rff_common['test_acc'], label='Test Acc RFF', marker='s')
    # QORC
    plt.errorbar(
        grouped_qorc["n_features"],
        grouped_qorc["mean_train_acc"],
        yerr=grouped_qorc["std_train_acc"],
        label="Train Acc QORC",
        marker="o",
        linestyle="-",
        capsize=5,
    )
    plt.errorbar(
        grouped_qorc["n_features"],
        grouped_qorc["mean_test_acc"],
        yerr=grouped_qorc["std_test_acc"],
        label="Test Acc QORC",
        marker="o",
        linestyle="--",
        capsize=5,
    )

    # RFF
    plt.errorbar(
        grouped_rff["n_features"],
        grouped_rff["mean_train_acc"],
        yerr=grouped_rff["std_train_acc"],
        label="Train Acc RFF",
        marker="s",
        linestyle="-",
        capsize=5,
    )
    plt.errorbar(
        grouped_rff["n_features"],
        grouped_rff["mean_test_acc"],
        yerr=grouped_rff["std_test_acc"],
        label="Test Acc RFF",
        marker="s",
        linestyle="--",
        capsize=5,
    )

    # Ajouter des labels et une légende
    plt.xlabel("n_features")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracies vs n_features")
    plt.ylim(0.88, 1.00)  # Ajustement de l'échelle Y pour voir 1.00
    plt.xticks(features_scale)
    plt.legend(loc="lower right")
    plt.grid(True)

    if len(f_out_img) > 2:
        dossier_parent = os.path.dirname(f_out_img)
        os.makedirs(dossier_parent, exist_ok=True)
        plt.savefig(f_out_img)
        print("Saved file:", f_out_img)

    # Afficher le graphe
    plt.show()


##########################################################
# Main script

if __name__ == "__main__":
    f_in_qorc_aggregated_results_csv = "results/f_out_results_training_qorc.csv"
    f_in_rff_aggregated_results_csv = "results/f_out_results_training_rff.csv"

    outdir = "outdir"
    # outdir = "outdir_ScaleWay/"
    aggregate_results_csv_files(outdir, f_in_rff_aggregated_results_csv)

    f_out_img = "results/graph_qorc_vs_rff.png"

    features_scale = list(range(2000, 6001, 2000))
    figsize_list = [8, 6]

    draw_graph_qorc_vs_rff(
        f_in_qorc_aggregated_results_csv,
        f_in_rff_aggregated_results_csv,
        features_scale,
        figsize_list,
        f_out_img,
    )
