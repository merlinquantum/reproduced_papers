#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import torch
import perceval as pcvl

from perceval.runtime import RemoteConfig
from merlin.core.merlin_processor import MerlinProcessor


def _spin_until_with_ctrlc(
    pred, timeout_s: float = 10.0, sleep_s: float = 0.02
) -> bool:
    import time as _t

    start = _t.time()
    try:
        while not pred():
            if _t.time() - start > timeout_s:
                return False
            _t.sleep(sleep_s)
        return True
    except KeyboardInterrupt:
        return False


def forward_remote_qorc_quantum_layer(
    train_tensor,
    val_tensor,
    test_tensor,
    qorc_quantum_layer,
    qpu_device_name,
    qpu_device_nsample,
    logger,
):
    # qpu_device_timeout = 12
    qpu_device_timeout = 60000  # roughly 15h

    # chunk_concurrency = 10
    chunk_concurrency = 20

    # max_batch_size    = 32
    # max_batch_size    = 64
    # max_batch_size    = 128
    # max_batch_size    = 1024
    # max_batch_size    = 10240  # 10k images in a row => Takes more time
    max_batch_size = 102400  # 100k images in a row => only one batch

    logger.info("Call to remote_qorc_quantum_layer ")
    logger.info(
        "Using MerlinProcessor with remote_processor name: {}".format(qpu_device_name)
    )
    # logger.info("max_batch_size:{}".format(max_batch_size))

    qpu_device_name = qpu_device_name.lower()

    LOCAL_STR = ":local"
    if LOCAL_STR in qpu_device_name:
        qorc_quantum_layer.force_simulation = (
            True  # Force local computation of the Quantum Layer
        )
        qpu_device_name = qpu_device_name.replace(LOCAL_STR, "")
        logger.info(
            "'{}' detected: local treatment of remote processor".format(LOCAL_STR)
        )

    valid_qpu_device_name_list = [
        "sim:slos",
        "sim:ascella",
        "sim:belenos",
        "qpu:ascella",
        "qpu:belenos",
    ]

    if qpu_device_name not in valid_qpu_device_name_list:
        logger.info(
            "Error in qorc: remote_processor_type not recognized:{}".format(
                qpu_device_name
            )
        )
        raise
        return -1

    # Création du MerlinProcessor
    qorc_quantum_layer.eval()
    token = os.environ.get("QUANDELA_TOKEN", "").strip()
    RemoteConfig.set_token(token)
    remote_processor = pcvl.RemoteProcessor(qpu_device_name)
    proc = MerlinProcessor(
        remote_processor,
        chunk_concurrency=chunk_concurrency,
        microbatch_size=max_batch_size,
    )

    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    test_size = test_tensor.shape[0]
    data_tensor = torch.cat([train_tensor, val_tensor, test_tensor], dim=0)
    logger.info("data_tensor.shape:{}".format(str(data_tensor.shape)))

    logger.info(
        f"Qorc: Call to forward async for remote processor: {qpu_device_name} - Compute train/val/test"
    )
    time_cour = time.time()
    fut = proc.forward_async(
        qorc_quantum_layer, data_tensor, nsample=qpu_device_nsample
    )
    _spin_until_with_ctrlc(
        lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=qpu_device_timeout
    )
    processed_data_tensor = fut.wait()
    duration = time.time() - time_cour
    logger.info(f"Duration (s): {duration}")

    train_data_qorc = processed_data_tensor[:train_size]
    val_data_qorc = processed_data_tensor[train_size : (train_size + val_size)]
    test_data_qorc = processed_data_tensor[-test_size:]

    return train_data_qorc, val_data_qorc, test_data_qorc
