#!/usr/bin/env python3

import math
import os
import random
import time

import merlin as ML
import numpy as np
import perceval as pcvl
import torch
from perceval.runtime import RemoteConfig
from perceval.utils import NoiseModel
from sklearn.decomposition import PCA
from torch import nn

from lib.lib_datasets import (
    get_dataloader,
    get_qorc_dataset,
    split_fold_numpy,
)
from lib.lib_learning import get_device, model_eval, model_fit


def get_circuit_physical_depth(circuit: pcvl.Circuit):
    t = type(circuit)
    match t:
        case pcvl.components.BS:
            return 1, [1]
        case pcvl.components.PS:
            return 0, [0]
        case pcvl.components.Unitary:
            return 2 * circuit.m, [2, 2]
        case pcvl.components.Circuit:
            if circuit.is_composite():
                depths = [0] * circuit.m
                d_current = 0
                for modes, comp in circuit._components:  # type: ignore[attr-defined]
                    # print(modes, comp)
                    d_current = max(depths[m] for m in modes)
                    add_depth, _ = get_circuit_physical_depth(comp)
                    for m in modes:
                        depths[m] = d_current + add_depth
                d_current = max(depths[m] for m in modes)
                return d_current, depths
            else:
                raise ValueError(
                    "Erreur dans get_circuit_physical_depth: Le circuit n'est pas composite."
                )
        case _:
            raise ValueError(
                f"Erreur dans get_circuit_physical_depth: Type de circuit non géré: {t}"
            )
    raise ValueError("Erreur dans get_circuit_physical_depth (interne).")


def get_PS_name_for_mode_and_depth(circuit: pcvl.Circuit, mode: int, depth: int):
    if not circuit.is_composite():
        raise ValueError("Erreur: Circuit pas composite")

    depths = [0] * circuit.m
    for modes, comp in circuit._components:  # type: ignore[attr-defined]
        # print(modes, comp)
        d_current = max(depths[m] for m in modes)

        add_depth = None
        if isinstance(comp, pcvl.components.BS):
            add_depth = 1
        if isinstance(comp, pcvl.components.PS):
            add_depth = 0
        if add_depth is None:
            raise ValueError("Erreur: Composant non reconnu")

        for m in modes:
            depths[m] = d_current + add_depth

        if (
            isinstance(comp, pcvl.components.PS)
            and mode in modes
            and depths[mode] >= depth
        ):
            ps_name = comp.get_variables()["phi"]
            return ps_name, depths[mode]

    # Pas de Phaseshifter trouvé avec une profondeur en BS suffisante (la depth demandée est trop élevée pour le circuit)
    return None, None


def create_quantum_layer_for_ascella(n_photons, logger):
    run_seed = 24
    n_modes = 12

    token = os.environ.get("QUANDELA_TOKEN", "").strip()
    RemoteConfig.set_token(token)
    remote_processor = pcvl.RemoteProcessor("sim:ascella")

    specs = remote_processor.specs
    spec_circuit = specs["specific_circuit"]
    d_current, depths = get_circuit_physical_depth(spec_circuit)
    print("circuit depths:", d_current, depths)

    # Ascella: On cherche les PS du milieu, pour les 11 derniers modes, car le premier mode n'a pas de PhaseShifter
    input_param_names = []
    for mode_cour in range(1, 12):
        depth_target = depths[mode_cour] // 2
        ps_name, depth_cour = get_PS_name_for_mode_and_depth(
            spec_circuit, mode_cour, depth_target
        )
        print(mode_cour, depth_target, depth_cour, ps_name)
        input_param_names.append(ps_name)
    print("Liste des paramètres d'input:", input_param_names)

    # On construit un circuit identique, avec des phases fixes pour les non-input
    qorc_circuit = pcvl.Circuit(n_modes)
    np.random.seed(run_seed)
    for modes, comp in spec_circuit._components:  # type: ignore[attr-defined]
        if isinstance(comp, pcvl.components.BS):
            qorc_circuit.add(modes, comp)
        if isinstance(comp, pcvl.components.PS):
            ps_name = comp.get_variables()["phi"]
            if ps_name in input_param_names:
                qorc_circuit.add(modes, comp)
            if ps_name not in input_param_names:
                phase = np.random.uniform(0, 2 * np.pi)
                qorc_circuit.add(modes, pcvl.components.PS(phase))

    logger.info("MerLin QuantumLayer creation:")
    qorc_output_size = math.comb(n_photons + n_modes - 1, n_photons)

    assert n_photons <= n_modes, (
        "Error with photons_input_mode: Bunching not possible for input state."
    )
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    qorc_input_state = [0] * n_modes
    for k in range(n_photons):
        index = round(k * step)
        qorc_input_state[index] = 1

    device_name = "cpu"
    qorc_quantum_layer = ML.QuantumLayer(
        input_size=n_modes
        - 1,  # Nb input features = 11 pour ascella (le premier mode n'a pas de PS)
        circuit=qorc_circuit,  # QORC quantum circuit
        trainable_parameters=[],  # Circuit is not trainable
        input_parameters=input_param_names,  # Input encoding parameters
        input_state=qorc_input_state,  # Initial photon state
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),  # Output: Get all Fock states probas
        device=torch.device(device_name),
    )

    # Verify there are no trainable parameters
    params = qorc_quantum_layer.parameters()
    count = sum(1 for _ in params)
    assert count == 0, f"quantum_layer does not have 0 parameters: {count}"

    logger.info("Created QuantumLayer:")
    logger.info(str(qorc_quantum_layer))

    return qorc_quantum_layer, qorc_output_size


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
        "Error with photons_input_mode: Bunching not possible for input state."
    )
    step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
    qorc_input_state = [0] * n_modes
    for k in range(n_photons):
        index = round(k * step)
        qorc_input_state[index] = 1

    params_prefix = ["px"]

    if b_no_bunching:
        qorc_output_size = math.comb(n_modes, n_photons)
    else:
        qorc_output_size = math.comb(n_photons + n_modes - 1, n_photons)
    strategy = (
        ML.MeasurementStrategy.probs()
        if b_no_bunching
        else ML.MeasurementStrategy.probs(computation_space=ML.ComputationSpace.FOCK)
    )
    logger.info("MerLin QuantumLayer creation:")
    qorc_quantum_layer = ML.QuantumLayer(
        input_size=n_modes,  # Nb input features = nb modes
        circuit=qorc_circuit,  # QORC quantum circuit
        trainable_parameters=[],  # Circuit is not trainable
        input_parameters=params_prefix,  # Input encoding parameters
        input_state=qorc_input_state,  # Initial photon state
        measurement_strategy=strategy,
        device=torch.device(device_name),
    )
    qorc_quantum_layer.eval()  # Put the layer in eval (do not compute gradiants)

    # Verify there are no trainable parameters
    params = qorc_quantum_layer.parameters()
    count = sum(1 for _ in params)
    assert count == 0, f"quantum_layer does not have 0 parameters: {count}"

    logger.info("Created QuantumLayer:")
    logger.info(str(qorc_quantum_layer))
    return [qorc_quantum_layer, qorc_output_size]


def create_qorc_reservoir_classifier(
    n_photons,
    n_components,
    seed,
    device_name,
    b_no_bunching,
    input_features=28 * 28,
    n_classes=10,
    noise=None,
):
    """Create the Merlin 0.4 reservoir classifier used by QORC.

    Parameters
    ----------
    n_photons : int
        Number of photons injected into the frozen reservoir.
    n_components : int
        Number of PCA components encoded by the reservoir.
    input_features : int
        Number of flattened input features in each image. Default value is 784.
    noise : perceval.utils.NoiseModel|None
        Perceval source noise model. If omitted, the reservoir is ideal.
        Default value is None.
    seed : int
        Seed used for the reservoir unitary and readout initialization.
    device_name : str
        Torch device used by the classifier.
    b_no_bunching : bool
        Whether the measurement output excludes photon bunching states.
    n_classes : int
        Number of output classes. Default value is 10.

    Returns
    -------
    merlin.ReservoirClassifier
        Frozen reservoir with a trainable linear readout.

    Raises
    ------
    ValueError
        If the classifier configuration is invalid.
    """
    reservoir = ML.ReservoirClassifier(
        in_features=input_features,
        out_features=n_classes,
        n_photons=n_photons,
        reduction=PCA(n_components=n_components),
        concatenate=True,
        cache=True,
        seed=seed,
        device=torch.device(device_name),
        dtype=torch.float32,
    )
    if b_no_bunching:
        reservoir.layer.measurement_strategy = ML.MeasurementStrategy.probs()
    else:
        reservoir.layer.measurement_strategy = ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        )
    if noise is not None:
        reservoir.layer.noise = noise
    return reservoir


def create_perceval_noise_model(
    enabled,
    indistinguishability=1.0,
    g2=0.0,
    g2_distinguishable=True,
):
    """Create the Perceval source noise model for a reservoir experiment.

    Parameters
    ----------
    enabled : bool
        Whether to enable Perceval noise. Default value is False.
    indistinguishability : float
        Probability that photons are indistinguishable, in [0, 1].
        Default value is 1.0.
    g2 : float
        Second-order intensity correlation at zero delay, in [0, 1].
        Default value is 0.0.
    g2_distinguishable : bool
        Whether photons generated by the g2 process are distinguishable.
        Default value is True.

    Returns
    -------
    perceval.utils.NoiseModel|None
        Configured Perceval noise model, or None when disabled.

    Raises
    ------
    ValueError
        If a noise probability is outside [0, 1].
    """
    if not enabled:
        return None
    for parameter_name, parameter_value in (
        ("indistinguishability", indistinguishability),
        ("g2", g2),
    ):
        if not 0 <= parameter_value <= 1:
            raise ValueError(f"{parameter_name} must be between 0 and 1.")
    return NoiseModel(
        indistinguishability=indistinguishability,
        g2=g2,
        g2_distinguishable=g2_distinguishable,
    )


def qorc_encoding_and_linear_training(
    # Main parameters
    n_photons,
    n_modes,
    seed,
    # Dataset parameters
    dataset_name,
    dataset_sampling,
    dataset_sample_count,
    dataset_samples_per_class,
    fold_index,
    n_fold,
    dataset_truncate,
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
    noise_enabled,
    noise_indistinguishability,
    noise_g2,
    noise_g2_distinguishable,
    device_name,
    qpu_device_name,
    qpu_device_nsample,
    run_dir,
    logger,
    return_history=False,
    save_weights=False,
):
    compute_device = get_device(device_name)

    n_components = n_modes
    if "ascella" in qpu_device_name:
        n_modes = 12
        n_components = 11  # Ascella first mode does not contain any phaseShifter -> 11 inputs instead of 12
        logger.info(
            "Warning: ascella architecture detectd in qpu_device_name. Forcing n_modes=12 and n_components=11."
        )
    if "belenos" in qpu_device_name:
        n_modes = 24
        n_components = 24
        logger.info(
            "Warning: ascella architecture detectd in qpu_device_name. Forcing n_modes=24 and n_components=24."
        )

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
        "Call to qorc_encoding_and_linear_training: n_photons={}, n_modes={}, n_components={}, run_seed={}, fold_index={}".format(
            n_photons, n_modes, n_components, run_seed, fold_index
        )
    )
    time_t1 = time.time()

    logger.info("Loading QORC data ({})".format(dataset_name))
    val_train_data, val_train_label, test_data, test_label = get_qorc_dataset(
        dataset_name,
        sampling=dataset_sampling,
        sample_count=dataset_sample_count,
        samples_per_class=dataset_samples_per_class,
        seed=run_seed,
    )
    val_train_data = (
        val_train_data.reshape(val_train_data.shape[0], -1).astype(np.float32) / 255.0
    )

    if n_fold == 0:
        train_label, train_data = val_train_label, val_train_data
        val_label, val_data = val_train_label, val_train_data
    else:
        val_label, val_data, train_label, train_data = split_fold_numpy(
            val_train_label, val_train_data, n_fold, fold_index, split_seed=run_seed
        )

    test_data = test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0

    if dataset_truncate > 0:
        # Only use the first images of datasets (i.e. truncate datasets to length = dataset_truncate)
        # for testing purpose
        train_data = train_data[:dataset_truncate]
        train_label = train_label[:dataset_truncate]
        val_data = val_data[:dataset_truncate]
        val_label = val_label[:dataset_truncate]
        test_data = test_data[:dataset_truncate]
        test_label = test_label[:dataset_truncate]

    n_pixels = train_data.shape[1]
    n_classes = int(max(np.max(train_label), np.max(test_label))) + 1

    logger.info("Datasets sizes:")
    logger.info(train_label.shape)  # (48000,)
    logger.info(train_data.shape)  # (48000, 784)
    logger.info(val_label.shape)  # (12000,)
    logger.info(val_data.shape)  # (12000, 784)
    logger.info(test_label.shape)  # (10000,)
    logger.info(test_data.shape)  # (10000, 784)

    ####################################################
    # Quantum reservoir feature computation
    logger.info("Creation of the Merlin ReservoirClassifier...")
    reservoir = create_qorc_reservoir_classifier(
        n_photons=n_photons,
        n_components=n_components,
        seed=run_seed,
        device_name=device_name,
        b_no_bunching=b_no_bunching,
        input_features=n_pixels,
        n_classes=n_classes,
        noise=create_perceval_noise_model(
            enabled=noise_enabled,
            indistinguishability=noise_indistinguishability,
            g2=noise_g2,
            g2_distinguishable=noise_g2_distinguishable,
        ),
    )

    logger.info("Computation of the quantum features...")
    time_t2 = time.time()
    remote_processor = None
    if qpu_device_name not in ("none", ""):
        from lib.lib_remote_qorc import create_remote_qorc_processor

        remote_processor = create_remote_qorc_processor(
            qpu_device_name, reservoir.layer, qpu_device_nsample, logger
        )

    reservoir.fit_reservoir(train_data, processor=remote_processor)
    train_dataset = reservoir.make_dataset(train_data, train_label)
    val_dataset = reservoir.make_dataset(
        val_data, val_label, processor=remote_processor
    )
    test_dataset = reservoir.make_dataset(
        test_data, test_label, processor=remote_processor
    )
    qorc_output_size = reservoir.layer.output_size
    logger.info("Quantum features size: {}".format(qorc_output_size))
    logger.info("Computation over.")
    time_t3 = time.time()

    ####################################################
    # Prepare structures (Dataset, DataLoader)
    # Datasets
    ds_train = train_dataset
    ds_val = val_dataset
    ds_test = test_dataset

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
    model = reservoir
    model.train()

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
        _val_loss_history,
        _val_accuracy_history,
        _duree_totale,
        best_val_epoch,
        test_loss_history,
        test_accuracy_history,
        best_state_dict,
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
        test_loader=test_loader if return_history else None,
        save_weights=save_weights,
    )

    logger.info("Training over.")
    n_train_epochs = len(train_loss_history)
    time_t4 = time.time()

    logger.info("Final evaluation (on test set)")
    if save_weights:
        best_state_dict = torch.load(
            os.path.join(run_dir, f_out_weights), map_location=compute_device
        )
    if best_state_dict is None:
        raise RuntimeError("Training did not produce a best model state.")

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

    result = [
        train_acc,
        val_acc,
        test_acc,
        qorc_output_size,
        n_train_epochs,
        duration_qfeatures,
        duration_train,
        best_val_epoch,
    ]
    if return_history:
        test_predictions = []
        test_targets = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                test_predictions.extend(model(inputs).argmax(dim=1).cpu().tolist())
                test_targets.extend(targets.cpu().tolist())
        return {
            "summary": result,
            "train_accuracy": [float(value) for value in train_accuracy_history],
            "train_loss": [float(value) for value in train_loss_history],
            "test_loss": [float(value) for value in test_loss_history],
            "test_accuracy": [float(value) for value in test_accuracy_history],
            "test_predictions": test_predictions,
            "test_targets": test_targets,
        }
    return result
