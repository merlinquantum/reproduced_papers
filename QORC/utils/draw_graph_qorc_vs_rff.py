#!/usr/bin/env python3

# $ micromamba activate qml-cpu
# $ python utils/draw_graph_qorc_vs_rff.py

##########################################################
# Librairies loading and functions definitions

import os
import matplotlib.pyplot as plt
import pandas as pd


# Aggregate rff results from outdir to single csv file
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
    b_train_acc,
    b_test_acc,
    f_out_img,
):
    df_qorc = pd.read_csv(f_in_qorc_aggregated_results_csv)
    df_rff = pd.read_csv(f_in_rff_aggregated_results_csv)

    df_qorc = df_qorc.rename(columns={"qorc_output_size": "n_features"})
    df_rff = df_rff.rename(columns={"n_rff_features": "n_features"})

    # Filter df1 to keep only rows where n_photons == 3
    df_qorc_filt = df_qorc[df_qorc["n_photons"] == 3]

    # Filtrer df1 pour ne garder que les lignes avec n_photons == 3
    common_features = set(df_qorc_filt["n_features"]).intersection(
        set(df_rff["n_features"])
    )

    # Filter both dataframes to keep only common values
    df_qorc_common = df_qorc_filt[df_qorc_filt["n_features"].isin(common_features)]
    df_rff_common = df_rff[df_rff["n_features"].isin(common_features)]

    # Compute avg and StDev per n_feature, for qorc
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

    # Compute avg and StDev per n_feature, for RFF
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

    figsize = (figsize_list[0], figsize_list[1])
    plt.figure(figsize=figsize)

    # QORC curves
    if b_train_acc:
        plt.errorbar(
            grouped_qorc["n_features"],
            grouped_qorc["mean_train_acc"],
            yerr=grouped_qorc["std_train_acc"],
            label="Train Acc QORC",
            marker="o",
            linestyle="-",
            capsize=5,
        )
    if b_test_acc:
        plt.errorbar(
            grouped_qorc["n_features"],
            grouped_qorc["mean_test_acc"],
            yerr=grouped_qorc["std_test_acc"],
            label="Test Acc QORC",
            marker="o",
            linestyle="--",
            capsize=5,
        )

    # RFF curves
    if b_train_acc:
        plt.errorbar(
            grouped_rff["n_features"],
            grouped_rff["mean_train_acc"],
            yerr=grouped_rff["std_train_acc"],
            label="Train Acc RFF",
            marker="s",
            linestyle="-",
            capsize=5,
        )
    if b_test_acc:
        plt.errorbar(
            grouped_rff["n_features"],
            grouped_rff["mean_test_acc"],
            yerr=grouped_rff["std_test_acc"],
            label="Test Acc RFF",
            marker="s",
            linestyle="--",
            capsize=5,
        )

    plt.xlabel("n_features")
    plt.ylabel("Accuracy")
    s_title = "Accuracy vs n_features"
    if b_test_acc:
        s_title = "Test " + s_title
    if b_train_acc:
        s_title = "Train " + s_title
    plt.title(s_title)
    plt.ylim(0.88, 1.00)
    plt.xticks(features_scale)
    plt.legend(loc="lower right")
    plt.grid(True)

    if len(f_out_img) > 2:
        dossier_parent = os.path.dirname(f_out_img)
        os.makedirs(dossier_parent, exist_ok=True)
        plt.savefig(f_out_img)
        print("Saved file:", f_out_img)

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

    b_train_acc = False
    # b_train_acc = True
    # b_test_acc = False
    b_test_acc = True

    draw_graph_qorc_vs_rff(
        f_in_qorc_aggregated_results_csv,
        f_in_rff_aggregated_results_csv,
        features_scale,
        figsize_list,
        b_train_acc,
        b_test_acc,
        f_out_img,
    )
