#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module provides standard code for deep learning: train deep learning models, compute predictions over a given dataset.
Original authors: Vincent Espitalier <vincent.espitalier@cirad.fr>
                  Hervé Goëau <herve.goeau@cirad.fr>

Modified by: Vincent Espitalier <vincent.espitalier@quandela.com>
"""

import sys
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_device(device_name):
    return torch.device(device_name)


def affiche_tag_heure(logger):
    from datetime import datetime
    import time

    now = datetime.now()
    logger.info(
        "tag heure: {} ({})".format(now.strftime("%Y-%m-%d_%H:%M:%S"), time.time())
    )


def print_epoch_like_keras(epoch, n_epochs, logger):
    logger.info("Epoch {}/{}".format(epoch, n_epochs))


def print_lines_like_keras(n_batch, duration, epoch_loss, epoch_accuracy, logger):
    s_part1 = str(n_batch) + "/" + str(n_batch) + " - " + str(int(duration))
    s_part2 = "s - loss: {:.4f} - accuracy: {:.4f}".format(epoch_loss, epoch_accuracy)
    logger.info(s_part1 + s_part2)


def print_lines_like_keras_avec_validation(
    n_batch,
    duration,
    epoch_loss,
    epoch_accuracy,
    val_epoch_loss,
    val_epoch_accuracy,
    logger,
):
    s_part1 = str(n_batch) + "/" + str(n_batch) + " - " + str(int(duration))
    s_part2 = "s - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}".format(
        epoch_loss, epoch_accuracy, val_epoch_loss, val_epoch_accuracy
    )
    logger.info(s_part1 + s_part2)


def print_lines_like_keras_avec_validation_et_lr(
    n_batch,
    duration,
    epoch_loss,
    epoch_accuracy,
    val_epoch_loss,
    val_epoch_accuracy,
    lr,
    logger,
):
    s_part1 = str(n_batch) + "/" + str(n_batch) + " - " + str(int(duration))
    s_part2 = "s - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f} - lr: {:.4f}".format(
        epoch_loss, epoch_accuracy, val_epoch_loss, val_epoch_accuracy, lr
    )
    logger.info(s_part1 + s_part2)


def count_train_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Eval function with PyTorch
def model_eval(
    model, data_loader, criterion, device, logger, calc_accuracy=False, printPerf=True
):
    date_debut = time.perf_counter()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        running_loss = 0.0

        if calc_accuracy:
            running_corrects = 0.0

        n_verif_imgs = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if calc_accuracy:
                _, predictions = torch.max(outputs, 1)
                running_corrects += torch.sum(predictions == labels.data)
            n_verif_imgs += inputs.shape[0]

        epoch_loss = running_loss / len(data_loader.dataset)

        if calc_accuracy:
            epoch_accuracy = running_corrects.float() / len(data_loader.dataset)
        else:
            epoch_accuracy = -1

        date_fin = time.perf_counter()
        duree_epoch = date_fin - date_debut

        if printPerf:
            print_lines_like_keras(
                n_batch=len(data_loader),
                duration=duree_epoch,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                logger=logger,
            )

    return [epoch_loss, epoch_accuracy, duree_epoch]


def model_fit_train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    logger,
    calc_accuracy=False,
    printPerf=False,
):
    date_debut = time.perf_counter()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.train()

    running_loss = 0.0
    if calc_accuracy:
        running_corrects = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # Forward pass: Evaluation of the cost function
        loss = criterion(outputs, labels)
        if calc_accuracy:
            _, predictions = torch.max(outputs, 1)

        # loss = criterion(outputs, labels)

        # Reset optimizer gradiants to zero
        optimizer.zero_grad()

        # Backward pass: Compute gradiants (for params such that requires_grad = True.)
        loss.backward()

        # Update params values, by applying gradiant retropropagation
        optimizer.step()

        # _, predictions = torch.max(outputs, 1)

        # running_loss     += loss.item()
        running_loss += loss
        if calc_accuracy:
            running_corrects += torch.sum(predictions == labels.data)

    epoch_loss = running_loss.item() / len(data_loader.dataset)

    if calc_accuracy:
        epoch_accuracy = running_corrects.float() / len(data_loader.dataset)
    else:
        epoch_accuracy = -1

    date_fin = time.perf_counter()
    duree_epoch = date_fin - date_debut

    if printPerf:
        print_lines_like_keras(
            n_batch=len(data_loader),
            duration=duree_epoch,
            epoch_loss=epoch_loss,
            epoch_accuracy=epoch_accuracy,
            logger=logger,
        )

    return [epoch_loss, epoch_accuracy, duree_epoch]


def ReduceLROnPlateau_reduceLR(epoch, optimizer, reduce_lr_factor, logger):
    optimizer_state_dict = optimizer.state_dict()
    lr_curr = optimizer_state_dict["param_groups"][0]["lr"]
    lr_curr = lr_curr * reduce_lr_factor
    optimizer_state_dict["param_groups"][0]["lr"] = lr_curr
    optimizer.load_state_dict(optimizer_state_dict)
    logger.info(
        "Epoch {:05d}: ReduceLROnPlateau reducing learning rate to {:e}".format(
            epoch, lr_curr
        )
    )


# PyTorch implementation of the model learning
def model_fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    n_epochs,
    filename_model_weights_out,
    early_stop_patience,
    early_stop_min_delta,
    reduce_lr_patience,
    reduce_lr_factor,
    device,
    logger,
    b_use_cosine_scheduler=False,
    tf_train_writer=None,
    tf_val_writer=None,
    calc_accuracy=False,
):
    logger.info(
        "Call model_fit(), with {} parameters to train.".format(
            count_train_parameters(model)
        )
    )

    if b_use_cosine_scheduler:
        logger.info("Warning: Use cosine scheduler -> Disable reduce LR on plateau.")
        reduce_lr_patience = n_epochs + 1
        reduce_lr_factor = 1.0

    duration = 0

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_val_epoch_loss_checkpoint = float("inf")
    best_val_epoch_loss_earlystopping = float("inf")

    early_stop_n_epoch = 0
    reduce_lr_n_epoch = 0

    best_val_epoch = -1

    if b_use_cosine_scheduler:
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler = CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=curr_lr / 10.0
        )  # Le CosineAnnealing donne un LR divisé par 10 à la fin
        logger.info("Cosine scheduler activated")

    for epoch in range(1, n_epochs + 1):
        affiche_tag_heure(logger)
        sys.stdout.flush()

        # Learning: Performs one epoch of training with gradiant retropropagation algorithm
        [train_epoch_loss, train_epoch_accuracy, train_duree_epoch] = (
            model_fit_train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                logger,
                calc_accuracy,
                printPerf=False,
            )
        )

        # Evaluation on val_set
        [val_epoch_loss, val_epoch_acc, val_duree_epoch] = model_eval(
            model, val_loader, criterion, device, logger, calc_accuracy, printPerf=False
        )

        if tf_train_writer is not None:
            tf_train_writer.add_scalar("loss", train_epoch_loss, epoch)
            tf_train_writer.add_scalar("accuracy", train_epoch_accuracy, epoch)
        if tf_val_writer is not None:
            tf_val_writer.add_scalar("loss", val_epoch_loss, epoch)
            tf_val_writer.add_scalar("accuracy", val_epoch_acc, epoch)

        # Save metrics
        train_loss_history.append(train_epoch_loss)
        train_accuracy_history.append(train_epoch_accuracy)
        val_loss_history.append(val_epoch_loss)
        val_accuracy_history.append(val_epoch_acc)

        duration += train_duree_epoch + val_duree_epoch

        # Print results
        print_epoch_like_keras(epoch, n_epochs, logger)

        if b_use_cosine_scheduler:
            print_lines_like_keras_avec_validation_et_lr(
                n_batch=len(train_loader),
                duration=train_duree_epoch + val_duree_epoch,
                epoch_loss=train_epoch_loss,
                epoch_accuracy=train_epoch_accuracy,
                val_epoch_loss=val_epoch_loss,
                val_epoch_accuracy=val_epoch_acc,
                lr=curr_lr,
                logger=logger,
            )
            scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

        else:
            print_lines_like_keras_avec_validation(
                n_batch=len(train_loader),
                duration=train_duree_epoch + val_duree_epoch,
                epoch_loss=train_epoch_loss,
                epoch_accuracy=train_epoch_accuracy,
                val_epoch_loss=val_epoch_loss,
                val_epoch_accuracy=val_epoch_acc,
                logger=logger,
            )

        if val_epoch_loss < best_val_epoch_loss_checkpoint:
            # val_loss decreased: Save model
            filename_model_weights_out_checkpoint = filename_model_weights_out
            torch.save(model.state_dict(), filename_model_weights_out_checkpoint)
            logger.info(
                "Epoch {:05d}: val_loss improved from {:.5f} to {:.5f}, saving model to ".format(
                    epoch, best_val_epoch_loss_checkpoint, val_epoch_loss
                )
                + filename_model_weights_out_checkpoint
            )
            best_val_epoch_loss_checkpoint = val_epoch_loss
            best_val_epoch = epoch
            reduce_lr_n_epoch = 0
        else:
            logger.info(
                "Epoch {:05d}: val_loss did not improve from {:.5f}".format(
                    epoch, best_val_epoch_loss_checkpoint
                )
            )
            reduce_lr_n_epoch = reduce_lr_n_epoch + 1
            if reduce_lr_n_epoch >= reduce_lr_patience:
                ReduceLROnPlateau_reduceLR(epoch, optimizer, reduce_lr_factor, logger)
                reduce_lr_n_epoch = 0

        # Early Stopping: val_loss decreased by at least the quantity "early_stop_min_delta"
        if val_epoch_loss < best_val_epoch_loss_earlystopping - early_stop_min_delta:
            best_val_epoch_loss_earlystopping = val_epoch_loss
            early_stop_n_epoch = 0
        else:
            early_stop_n_epoch = early_stop_n_epoch + 1
            if early_stop_n_epoch >= early_stop_patience:
                logger.info("Epoch {:05d}: early stopping".format(epoch))
                break

        # Force printing outputs
        sys.stdout.flush()

    return [
        train_loss_history,
        train_accuracy_history,
        val_loss_history,
        val_accuracy_history,
        duration,
        best_val_epoch,
    ]
