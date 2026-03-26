# QEmbedding

This repository contains the implementation of Neural Quantum Embedding (NQE) as described in the paper [Neural Quantum Embedding: Pushing the Limits of Quantum Supervised Learning](https://arxiv.org/pdf/2311.11412). The project demonstrates how quantum machine learning (QML) performance can be enhanced through pre-trained quantum embeddings.

## Overview

Neural Quantum Embedding implements a novel approach to quantum machine learning that combines classical neural networks with quantum circuits. The key innovation is the optimization of quantum embedding circuit to enhance data distinguishability (or maximize trace distance).

## Features

- Implementation of Neural Quantum Embedding (NQE) models
- Compare Quantum Convolutional Neural Networks (QCNN) performances with and without NQE.
- Support for multiple datasets (MNIST, Fashion-MNIST, KMNIST)
- Experiments in different environments:
  - Noiseless simulation
  - Noisy simulation (using IBMQ fake backends)
  - Real quantum hardware (IBM Quantum)

## Project Structure

- `data.py`: Dataset loading and preprocessing
  - Supports MNIST, Fashion-MNIST, and KMNIST
  - Implements various feature reduction methods
  
- `embedding.py`: Quantum embedding implementations
  - ZZ feature embedding
  - Quantum Convolutional Neural Networks (QCNN)
  
- `Hybrid_nn.py`: Neural Quantum Embedding (NQE) models
  - Various hybrid classical-quantum architectures
  - Pre-training and fine-tuning implementations
  
- `training.py`: Training utilities for NQE models
  - Training loops
  - Optimization methods
  - Evaluation metrics

### Experimental Results

- `QCNN_demonstration/`: Contains results demonstrating NQE effectiveness
  - Noiseless simulations
  - Noisy simulations (using IBMQ fake backends)
  - Real hardware experiments (IBM Quantum)
  
- `Additional Experiments/`: Extended experiments on different datasets
  - Fashion-MNIST experiments
  - KMNIST experiments
  - Noisy and noiseless simulations

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hur2024neural,
  title={Neural quantum embedding: Pushing the limits of quantum supervised learning},
  author={Hur, Tak and Araujo, Israel F and Park, Daniel K},
  journal={Physical Review A},
  volume={110},
  number={2},
  pages={022411},
  year={2024},
  publisher={APS}
}
```

## License

This project is licensed under the terms of the included LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue in the repository.
