#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

import perceval as pcvl
import merlin as ML
from merlin.datasets.mnist_digits import get_data_train_original, get_data_test_original

from lib.lib_datasets import (
    tensor_dataset,
    get_dataloader,
    split_fold_numpy,
)
from lib.lib_learning import get_device, model_eval, model_fit


def create_qorc_quantum_layer(
    n_photons,  # Nb photons
    n_modes,  # Nb modes
    b_no_bunching,
    device_name,
    logger,
):
    logger.info(
        "Call to create_qorc_quantum_layer: {}, {}, {}, {}".format(
            n_photons, n_modes, b_no_bunching, device_name
        )
    )

    unitary = pcvl.Matrix.random_unitary(n_modes)  # Haar-uniform unitary sampling
    interferometer_1 = pcvl.Unitary(unitary)
    interferometer_2 = interferometer_1.copy()

    # Input Phase Shifters
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_modes):
        px = pcvl.P(f"px{i + 1}")
        port_range = i
        c_var.add(port_range, pcvl.PS(px))

    qorc_circuit = interferometer_1 // c_var // interferometer_2

    assert n_photons <= n_modes, (
        "Error with photons_input_mode: Too many photons versus modes with 'distributed'."
    )
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    qorc_input_state = [0] * n_modes
    for k in range(n_photons):
        index = int(round(k * step))
        qorc_input_state[index] = 1

    params_prefix = ["px"]

    if b_no_bunching:
        qorc_output_size = math.comb(n_modes, n_photons)
    else:
        qorc_output_size = math.comb(n_photons + n_modes - 1, n_photons)

    logger.info("MerLin QuantumLayer creation:")
    qorc_quantum_layer = ML.QuantumLayer(
        input_size=n_modes,  # Nb input features = nb modes
        output_size=qorc_output_size,  # Nb output classes = nb modes
        circuit=qorc_circuit,  # QORC quantum circuit
        trainable_parameters=[],  # Circuit is not trainable
        input_parameters=params_prefix,  # Input encoding parameters
        input_state=qorc_input_state,  # Initial photon state
        output_mapping_strategy=ML.OutputMappingStrategy.NONE,  # Output: Get all Fock states probas
        # See: https://merlinquantum.ai/user_guide/output_mappings.html
        no_bunching=b_no_bunching,
        device=torch.device(device_name),
    )

    # Verify there are no trainable parameters
    params = qorc_quantum_layer.parameters()
    count = sum(1 for _ in params)
    assert count == 0, f"quantum_layer does not have 0 parameters: {count}"

    logger.info("Created QuantumLayer:")
    logger.info(str(qorc_quantum_layer))
    return [qorc_quantum_layer, qorc_output_size]


def qorc_encoding_and_linear_training(
    # Main parameters
    n_photons,
    n_modes,
    seed,
    # Dataset parameters
    fold_index,
    n_fold,
    # Training parameters
    n_epochs,
    batch_size,
    learning_rate,
    reduce_lr_patience,
    reduce_lr_factor,
    num_workers,
    pin_memory,
    f_out_weights,
    # Other parameters
    b_no_bunching,
    b_use_tensorboard,
    device_name,
    run_dir,
    logger,
):
    storage_device = torch.device("cpu")
    compute_device = get_device(device_name)

    run_seed = seed
    if run_seed >= 0:
        # Seeding to control the random generators
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(seed=run_seed)
        torch.cuda.manual_seed_all(seed=run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(mode=False)

    logger.info(
        "Call to qorc_encoding_and_linear_training: n_photons={}, n_modes={}, run_seed={}, fold_index={}".format(
            n_photons, n_modes, run_seed, fold_index
        )
    )
    time_t1 = time.time()

    logger.info("Loading MNIST data...")
    val_train_data, val_train_label, _ = get_data_train_original()
    val_train_data = (
        val_train_data.reshape(val_train_data.shape[0], -1).astype(np.float32) / 255.0
    )

    val_label, val_data, train_label, train_data = split_fold_numpy(
        val_train_label, val_train_data, n_fold, fold_index, split_seed=run_seed
    )

    test_data, test_label, _ = get_data_test_original()
    test_data = test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0
    n_pixels = 28 * 28  # MNIST images size
    n_classes = 10  # 10 classes, one for each figure

    logger.info("Datasets sizes:")
    logger.info(train_label.shape)  # (48000,)
    logger.info(train_data.shape)  # (48000, 784)
    logger.info(val_label.shape)  # (12000,)
    logger.info(val_data.shape)  # (12000, 784)
    logger.info(test_label.shape)  # (10000,)
    logger.info(test_data.shape)  # (10000, 784)

    ####################################################
    # Quantum features computation
    logger.info("Creation of the encoder of the quantum reservoir...")

    # 1) PCA Components computation
    pca = PCA(n_components=n_modes)
    train_data_pca = pca.fit_transform(train_data)
    val_data_pca = pca.transform(val_data)
    test_data_pca = pca.transform(test_data)

    # 2) PCA comp normalization (to [0, 1] (global min/max) )
    pca_train_global_min = train_data_pca.min()
    pca_train_global_max = train_data_pca.max()

    def normalize_global_min_max(data, global_min, global_max):
        epsilon = 1e-8  # Avoid zero division
        return (data - global_min) / (global_max - global_min + epsilon)

    train_data_pca_norm = normalize_global_min_max(
        train_data_pca, pca_train_global_min, pca_train_global_max
    )
    val_data_pca_norm = normalize_global_min_max(
        val_data_pca, pca_train_global_min, pca_train_global_max
    )
    test_data_pca_norm = normalize_global_min_max(
        test_data_pca, pca_train_global_min, pca_train_global_max
    )

    # 3) Qorc quantum layer creation
    [qorc_quantum_layer, qorc_output_size] = create_qorc_quantum_layer(
        n_photons,  # Nb photons
        n_modes,  # Nb modes
        b_no_bunching,
        device_name,
        logger,
    )

    logger.info("Quantum features size: {}".format(qorc_output_size))
    logger.info("Encoding of the PCA comps to quantum features...")
    time_t2 = time.time()
    train_data_qorc = qorc_quantum_layer(
        torch.tensor(train_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    val_data_qorc = qorc_quantum_layer(
        torch.tensor(val_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    test_data_qorc = qorc_quantum_layer(
        torch.tensor(test_data_pca_norm, dtype=torch.float32, device=compute_device)
    )
    logger.info("Encoding over.")
    time_t3 = time.time()

    # 4) Quantum features normalization (standard_scaler)
    qorc_train_mean = train_data_qorc.detach().mean(dim=0)
    qorc_train_std = train_data_qorc.detach().std(dim=0)

    def normalize_standard_scaler(data, mean, std):
        epsilon = 1e-8  # Avoid zero division
        return (data - mean) / (std + epsilon)

    train_data_qorc_norm = normalize_standard_scaler(
        train_data_qorc, qorc_train_mean, qorc_train_std
    )
    val_data_qorc_norm = normalize_standard_scaler(
        val_data_qorc, qorc_train_mean, qorc_train_std
    )
    test_data_qorc_norm = normalize_standard_scaler(
        test_data_qorc, qorc_train_mean, qorc_train_std
    )

    dtype = torch.float32
    all_train_data = torch.cat(
        (
            torch.tensor(train_data, dtype=dtype, device=compute_device),
            train_data_qorc_norm,
        ),
        dim=1,
    )
    all_val_data = torch.cat(
        (
            torch.tensor(val_data, dtype=dtype, device=compute_device),
            val_data_qorc_norm,
        ),
        dim=1,
    )
    all_test_data = torch.cat(
        (
            torch.tensor(test_data, dtype=dtype, device=compute_device),
            test_data_qorc_norm,
        ),
        dim=1,
    )

    ####################################################
    # Prepare structures (Dataset, DataLoader)
    # Datasets
    ds_train = tensor_dataset(
        all_train_data,
        train_label,
        storage_device,
        dtype=torch.float32,
        transform=None,
        n_side_pixels=28,
    )
    ds_val = tensor_dataset(
        all_val_data, val_label, storage_device, dtype=torch.float32
    )
    ds_test = tensor_dataset(
        all_test_data, test_label, storage_device, dtype=torch.float32
    )

    logger.info("train dataset len: {}".format(len(ds_train)))
    logger.info("val dataset len  : {}".format(len(ds_val)))
    logger.info("test dataset len : {}".format(len(ds_test)))

    # Dataloaders
    shuffle_train = True
    shuffle_test = True
    train_loader = get_dataloader(
        ds_train, batch_size, shuffle_train, num_workers, pin_memory, run_seed
    )
    val_loader = get_dataloader(
        ds_val, batch_size, shuffle_test, num_workers, pin_memory, run_seed
    )
    test_loader = get_dataloader(
        ds_test, batch_size, shuffle_test, num_workers, pin_memory, run_seed
    )

    logger.info("train loader len: {}".format(len(train_loader)))
    logger.info("val loader len  : {}".format(len(val_loader)))
    logger.info("test loader len : {}".format(len(test_loader)))

    ####################################################
    # Prepare the model and structures for training
    logger.info("Prepare the linear classifier")

    n_model_input_features = n_pixels + qorc_output_size
    logger.info("n_model_input_features: {}".format(n_model_input_features))
    linear = nn.Linear(
        n_model_input_features, n_classes, bias=True, device=compute_device
    )

    nn.init.xavier_uniform_(linear.weight)  # Xavier uniforme init (Glorot)
    nn.init.zeros_(linear.bias)
    model = linear
    model.to(compute_device)
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    logger.info("Evaluation before training (on test set)")
    calc_accuracy = True
    printPerf = True
    _eval_test = model_eval(
        model, test_loader, criterion, compute_device, logger, calc_accuracy, printPerf
    )

    logger.info("Beginning of training")
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, eps=1e-7)

    if b_use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        xp_name = (
            str(n_photons)
            + "photons_"
            + str(n_modes)
            + "modes_"
            + str(run_seed)
            + "seed_"
            + str(fold_index)
            + "fold"
        )
        tf_train_writer = SummaryWriter(
            os.path.join(run_dir, "runs/" + xp_name + "_train")
        )
        tf_val_writer = SummaryWriter(os.path.join(run_dir, "runs/" + xp_name + "_val"))
    else:
        tf_train_writer = None
        tf_val_writer = None

    early_stop_patience = n_epochs
    early_stop_min_delta = 0.000001
    b_use_cosine_scheduler = False
    [
        train_loss_history,
        train_accuracy_history,
        val_loss_history,
        val_accuracy_history,
        duree_totale,
        best_val_epoch,
    ] = model_fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        n_epochs,
        os.path.join(run_dir, f_out_weights),
        early_stop_patience,
        early_stop_min_delta,
        reduce_lr_patience,
        reduce_lr_factor,
        compute_device,
        logger,
        b_use_cosine_scheduler,
        tf_train_writer=tf_train_writer,
        tf_val_writer=tf_val_writer,
        calc_accuracy=calc_accuracy,
    )

    logger.info("Training over.")
    n_train_epochs = len(train_loss_history)
    time_t4 = time.time()

    logger.info("Final evaluation (on test set)")
    best_state_dict = torch.load(
        os.path.join(run_dir, f_out_weights), map_location=compute_device
    )

    try:
        model.load_state_dict(best_state_dict)
        logger.info("n_model_input_features: {n_model_input_features}")
        [_, train_acc, _] = model_eval(
            model,
            train_loader,
            criterion,
            compute_device,
            logger,
            calc_accuracy,
            printPerf,
        )
        train_acc = int(1000000.0 * train_acc.item()) / 1000000.0
        [_, val_acc, _] = model_eval(
            model,
            val_loader,
            criterion,
            compute_device,
            logger,
            calc_accuracy,
            printPerf,
        )
        val_acc = int(1000000.0 * val_acc.item()) / 1000000.0
        [_, test_acc, _] = model_eval(
            model,
            test_loader,
            criterion,
            compute_device,
            logger,
            calc_accuracy,
            printPerf,
        )
        test_acc = int(1000000.0 * test_acc.item()) / 1000000.0
    except RuntimeError as e:
        logger.info(f"Error while loading state_dict : {e}")
        train_acc = float("nan")
        val_acc = float("nan")
        test_acc = float("nan")
    time_t5 = time.time()

    duration_creation_couche_quantique = int(100.0 * (time_t2 - time_t1)) / 100.0
    logger.info(
        "Duration - Quantum layer creation: {}s".format(
            duration_creation_couche_quantique
        )
    )
    duration_calcul_quantum_features = int(100.0 * (time_t3 - time_t2)) / 100.0
    logger.info(
        "Duration - Quantum features encoding: {}s".format(
            duration_calcul_quantum_features
        )
    )
    duration_qfeatures = (
        duration_creation_couche_quantique + duration_calcul_quantum_features
    )
    duration_train = int(100.0 * (time_t4 - time_t3)) / 100.0
    logger.info("Duration - training: {}s".format(duration_train))
    duration_totale = int(100.0 * (time_t5 - time_t1)) / 100.0
    logger.info("Duration - total: {}s".format(duration_totale))
    logger.info("Best val epoch: {}".format(best_val_epoch))

    return [
        train_acc,
        val_acc,
        test_acc,
        qorc_output_size,
        n_train_epochs,
        duration_qfeatures,
        duration_train,
        best_val_epoch,
    ]
