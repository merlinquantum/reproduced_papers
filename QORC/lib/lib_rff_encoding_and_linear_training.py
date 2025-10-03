#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from merlin.datasets.mnist_digits import get_data_train_original, get_data_test_original


def rff_encoding_and_linear_training(
    # Main parameters
    n_rff_features,
    sigma,
    regularization_c,
    seed,
    b_optim_via_sgd,
    max_iter_sgd,
    # Dataset parameters
    run_dir,
    logger,
):
    run_seed = seed
    if run_seed >= 0:
        # Seeding to control the random generators
        random.seed(run_seed)
        np.random.seed(run_seed)

    logger.info(
        "Call to rff_encoding_and_linear_training: n_rff_features={}, sigma={}, regularization_c={}, seed={}, b_optim_via_sgd={}".format(
            n_rff_features, sigma, regularization_c, seed, b_optim_via_sgd
        )
    )
    time_t1 = time.time()

    logger.info("Loading MNIST data...")
    train_data, train_label, _ = get_data_train_original()
    train_data = train_data.reshape(train_data.shape[0], -1).astype(np.float32) / 255.0

    test_data, test_label, _ = get_data_test_original()
    test_data = test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0

    logger.info("Datasets sizes:")
    logger.info(train_label.shape)  # (60000,)
    logger.info(train_data.shape)  # (60000, 784)
    logger.info(test_label.shape)  # (10000,)
    logger.info(test_data.shape)  # (10000, 784)

    # MNIST data normalization
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    time_t2 = time.time()

    # RFF features computation
    rff = RBFSampler(
        gamma=1.0 / (2 * sigma**2), n_components=n_rff_features, random_state=run_seed
    )
    rff.fit(scaled_train_data)
    train_data_rff = rff.transform(scaled_train_data)
    test_data_rff = rff.transform(scaled_test_data)

    time_t3 = time.time()

    # Build full training dataset (aggregate MNIST + RFF features)
    all_train_data = np.hstack(
        [scaled_train_data, train_data_rff]
    )  # Shape: [n_samples, 28x28 + n_rff_features]
    all_test_data = np.hstack([scaled_test_data, test_data_rff])

    scaler_rff = StandardScaler()
    scaler_rff.fit(all_train_data)
    scaled_all_train_data = scaler_rff.transform(all_train_data)
    scaled_all_test_data = scaler_rff.transform(all_test_data)

    # Linear SVC with/without SGD
    if b_optim_via_sgd:
        # Source: LeChat / MistralAI

        print("Fit de la SVM (via SGD et hinge loss)")
        n_samples = scaled_all_train_data.shape[0]
        clf = SGDClassifier(
            loss="hinge",  # Similar to LinearSVC
            alpha=1.0
            / (1.0 * n_samples * regularization_c),  # 1/(n_samples * C), with C=1.0
            max_iter=max_iter_sgd,
            tol=1e-2,  # tolerance
            random_state=run_seed,
            n_jobs=-1,  # parrallel computing if possible
        )

    else:
        print("Fit de la SVM (LinearSVC)")
        clf = LinearSVC(
            multi_class="ovr", random_state=run_seed, C=regularization_c
        )  # One-Versus-Rest: Train one binary classifier per class then select highest margin class

    clf.fit(scaled_all_train_data, train_label)
    train_model_pred = clf.predict(scaled_all_train_data)
    test_model_pred = clf.predict(scaled_all_test_data)

    time_t4 = time.time()

    train_acc = int(10000.0 * accuracy_score(train_label, train_model_pred)) / 10000.0
    test_acc = int(10000.0 * accuracy_score(test_label, test_model_pred)) / 10000.0

    duration_calcul_rff_features = int(100.0 * (time_t3 - time_t2)) / 100.0
    logger.info(
        "Duration - RFF features encoding: {}s".format(duration_calcul_rff_features)
    )
    duration_train = int(100.0 * (time_t4 - time_t3)) / 100.0
    logger.info("Duration - training: {}s".format(duration_train))
    duration_totale = int(100.0 * (time_t4 - time_t1)) / 100.0
    logger.info("Duration - total: {}s".format(duration_totale))

    return [
        train_acc,
        test_acc,
        duration_calcul_rff_features,
        duration_train,
    ]
