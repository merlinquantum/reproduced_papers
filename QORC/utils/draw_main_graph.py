#!/usr/bin/env python3

# $ micromamba activate qml-cpu
# $ python utils/draw_main_graph.py

##########################################################
# Librairies loading and functions definitions

import os
import matplotlib.pyplot as plt
import pandas as pd


# Aggregate qorc results from outdir to single csv file
def aggregate_results_csv_files(outdir, f_out_aggregated_csv):
    dataframes = []
    for root, _dirs, files in os.walk(outdir):
        for file in files:
            if "f_out_results_training_qorc" in file:
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


def draw_main_graph(
    f_in_aggregated_results_csv,
    figsize_list,
    b_train_acc,
    b_test_acc,
    x_min,
    x_max,
    y_min,
    y_max,
    f_out_img,
):
    df = pd.read_csv(f_in_aggregated_results_csv)

    disk_size_ratio = 0.3
    legend_disk_size_list = [800, 1600, 2400, 3200, 4000, 4800]
    figsize = (figsize_list[0], figsize_list[1])

    colors = plt.cm.tab10.colors
    s_qorc_output_size_name = "qorc_output_size"

    # Compute avg and StDev per couple (n_photons, n_modes)
    grouped = (
        df.groupby(["n_photons", "n_modes"])
        .agg(
            mean_train_acc=("train_acc", "mean"),
            std_train_acc=("train_acc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_qorc_output_size=("qorc_output_size", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=figsize)

    # train_acc curves
    if b_train_acc:
        for i, n_photon in enumerate(grouped["n_photons"].unique()):
            subset = grouped[grouped["n_photons"] == n_photon]
            color = colors[i % len(colors)]
            marker = "o"
            plt.errorbar(
                subset["n_modes"],
                subset["mean_train_acc"],
                yerr=subset["std_train_acc"],
                marker=marker,
                color=color,
                label=f"Train Acc (n_photons={n_photon})",
                linestyle="-",
                capsize=5,
            )

    # test_acc curves
    if b_test_acc:
        for i, n_photon in enumerate(grouped["n_photons"].unique()):
            subset = grouped[grouped["n_photons"] == n_photon]
            color = colors[i % len(colors)]
            marker = "o"
            plt.errorbar(
                subset["n_modes"],
                subset["mean_test_acc"],
                yerr=subset["std_test_acc"],
                marker=marker,
                color=color,
                label=f"Test Acc (n_photons={n_photon})",
                linestyle="--",
                capsize=5,
            )

        for _, row in grouped.iterrows():
            n_photon = row["n_photons"]
            i = list(grouped["n_photons"].unique()).index(n_photon)
            color = colors[i % len(colors)]
            plt.scatter(
                row["n_modes"],
                row["mean_test_acc"],
                s=row["mean_qorc_output_size"] * disk_size_ratio,
                color=color,
                alpha=0.30,
                edgecolors="none",
            )

    # Configure graph
    plt.xscale("log")
    plt.xlabel("n_modes (log scale)")
    plt.ylabel("Accuracy")
    s_title = "Accuracy vs n_modes"
    if b_test_acc:
        s_title = "Test " + s_title
    if b_train_acc:
        s_title = "Train " + s_title
    plt.title(s_title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, which="both", ls="--")

    # Legends
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    for size in legend_disk_size_list:
        plt.scatter(
            [],
            [],
            s=size * disk_size_ratio,
            color="gray",
            alpha=0.3,
            label=s_qorc_output_size_name + "=" + str(size),
        )
    plt.legend(bbox_to_anchor=(1.05, 0.9), loc="upper left", title="Disk Size", labelspacing=2.7)

    plt.tight_layout()

    if len(f_out_img) > 2:
        dossier_parent = os.path.dirname(f_out_img)
        os.makedirs(dossier_parent, exist_ok=True)
        plt.savefig(f_out_img)
        print("Saved file:", f_out_img)

    plt.show()


##########################################################
# Main script

if __name__ == "__main__":
    # outdir = "outdir"
    outdir = "outdir_ScaleWay/"
    f_in_aggregated_results_csv = "results/f_out_results_training_qorc.csv"
    aggregate_results_csv_files(outdir, f_in_aggregated_results_csv)

    f_out_img = "results/main_graph.png"

    figsize_list = [15, 9]

    b_train_acc = False
    # b_train_acc = True
    # b_test_acc = False
    b_test_acc = True

    [x_min, x_max] = [9.2, 220.0]
    [y_min, y_max] = [0.92, 1.0]

    draw_main_graph(
        f_in_aggregated_results_csv,
        figsize_list,
        b_train_acc,
        b_test_acc,
        x_min,
        x_max,
        y_min,
        y_max,
        f_out_img,
    )
