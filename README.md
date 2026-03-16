# MerLin Reproduced Papers

## About this repository


This repository contains implementations and resources for reproducing key quantum machine learning papers, with a focus on photonic and optical quantum computing.

It is part of the main MerLin project: [https://github.com/merlinquantum/merlin](https://github.com/merlinquantum/merlin)
and complements the online documentation available at:

[https://merlinquantum.ai/research/reproduced_papers.html](https://merlinquantum.ai/research/reproduced_papers.html)

Each paper reproduction is designed to be accessible, well-documented, and easy to extend. Contributions are welcome!


## Papers reproduced:
| Paper | Reproduction |
| --- | --- |
| [Quantum Optical Reservoir Computing](papers/QORC/). Sources: [munro_2024](https://opg.optica.org/abstract.cfm?uri=CLEO_FS-2024-FM2K.2), [rambach_2025](http://arxiv.org/abs/2512.08318), [sakurai_simple_2025](http://arxiv.org/abs/2405.14245), [lau_2025](http://arxiv.org/abs/2412.19336), [sakurai_2025](https://opg.optica.org/opticaqabstract.cfm?uri=opticaq-3-3-238) | - Scalability: increasing number of modes leads to better performance for the reservoir on MNIST.<br>- In the original work and in our reproduction, we see a quantum boost with modest resources.<br>- Our reproduction is QPU compliant. |
|[Computational Advantage in Hybrid Quantum NeuralNetworks: Myth or Reality?](papers/HQNN_MythOrReality/). Source: [kashif_2024](http://arxiv.org/abs/2412.04991)  | The original paper and our reproduction display that an HQNN model requires less parameters than a classical NN to achieve at least 90% accuracy on the noisy spiral dataset used that has a variable number of features (between 5 and 60). |
| [Quantum Self-Supervised Learning](papers/qSSL/). Source: [jaderberg_2021](https://arxiv.org/abs/2103.14653) | The MerLin model is better and faster than qiskit. On the first five classes of CIFAR-10 with two epochs and only eight modes for the QSSL: <br>- $\times0.97$ speedup versus a fully classical model compared to qiskit's $\times0.08$ speedup.<br>- Accuracy of 49.2% compared to 48.4% for qiskit and 48.1% for a classical SSL. 
| [Large-Language Model Fine-Tuning](papers/qLLM/). Source: [kim_2025](https://arxiv.org/abs/2504.08732) | The original paper states that the quantum enhanced model improves the accuracy to up to 3.14% compared to classical models with comparable number of parameters on a sentiment classification task on text data. On our end, all the best performing models (whether quantum or classical) reach around 89% accuracy without a clear segmentation. |
| [Quantum Long Short-Term Memory](papers/QLSTM/). Source:  [chen_2020](http://arxiv.org/abs/2009.01783) | Our MerLin-based photonic QLSTM yields similar results to the original gate-based QLSTM on function fitting tasks. However, the weaknesses of the classical LSTM reported in the paper were not fully present in our reproduction. |
| [Fock State-enhanced expressivity of Quantum Machine Learning Models](papers/fock_state_expressivity/). Source:  [gan_2022](https://arxiv.org/abs/2107.05224) | As explained and displayed in the original paper, our experiments also showcase that an increase in the number of photons used (when using the data encoding scheme that is proposed) is intrinsically linked to an increase in Variational Quantum Circuit (VQC) expressivity . |
| [Photonic Quantum Convolutional Neural Networks with Adaptive State Injection](papers/photonic_QCNN/). Source: [monbroussou_2025](https://arxiv.org/abs/2504.20989) | Our reproduction improved some of the reported accuracies for binary image classification using the proposed model by optimizing hyperparameters: <br>- Test accuracy on Custom BAS went from 92.7 ± 2.1 % to 98.2 ± 2.2 %.<br>- Test accuracy on MNIST (0 vs 1) went from 93.1 ± 3.6 % to 98.8 ± 1.0 %. |
| [Quantum Convolutional Neural Networks](papers/QCNN_data_classification/). Source [hur_2022](http://arxiv.org/abs/2108.00661) | The source paper reports that under similar training budgets, the QCNN outperforms the CNN. For linear optics, the interferometer constrains the number of trainable parameters therefore, we use more parameters as in the original work. However, our results show that our photonic QCNN outperforms the same CNNs as in the original paper, as well as CNN with similar number of kernels and kernel sizes. |
| [Quantum Relational Knowledge Distillation](papers/QRKD/). Source: [liu_2025](https://arxiv.org/abs/2508.13054) | In both the reference paper and in our reproduction, we see that the improvement of the student model due to the distillation is superior in the quantum relational knowledge distillation scheme compared to in its classical counterpart. |
| Quantum Recurrent Neural Networks for Sequential Learning. Source: [li_2023](https://arxiv.org/abs/2302.03244) | *Analysis to be done* |
| [Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing](papers/DQNN/). Source: [chen_2025](https://arxiv.org/abs/2505.08474) | The paper and our reproduction reach the conclusion that fewer quantum parameters need to be trained to obtain all the classical parameters. We also reach these following results:<br> - Accuracy better or worse with an approximate 2% error for bond dimensions in the training accuracy.<br>- Accuracy better or worse with an approximate 4% error for bond dimensions in the testing accuracy with 4 times less epochs.<br>- Speedup in the training using ADAM optimizer instead of COBYLA. Nonetheless, we attain different results for the ablation study. |
| [Data Reuploading](papers/data_reuploading/) [mauser_2025](http://arxiv.org/abs/2507.05120) | Our results confirm that the fully quantum data reuploading model is well performing and resource-efficient in the context of binary classification on the four datasets used. We also obtain that the model's expressivity scales with its number of reuplading layers. |
| [Nearest Centroids](papers/nearest_centroids_merlin/). Source: [johri_2020](https://arxiv.org/abs/2012.04145) | Reproducing this photon native algorithm led to accuracies that match the ones obtained classically on the three datasets of interest, as reported in the source paper. |
| Photonic QGAN. Source [sedrakyan_2024](https://opg.optica.org/opticaq/abstract.cfm?uri=opticaq-2-6-458)| The original code in Perceval and its implementation in MerLin yielded the same results (SSIM) with a training speed-up up to 15 times.|
| Quantum Enhanced Kernels. Source: [yin_2025](https://www.nature.com/articles/s41566-025-01682-5) | As in the original work, we show improved accuracy with the size of the training set and the geometric difference.|
| Quantum Transfer Learning. Source: [mari_2020](https://arxiv.org/abs/1912.08278) | Three transfer learning frameworks are reproduced: classical to classical, classical to quantum and quantum to classical. We obtain about the same accuracy with the MerLin implementation and the simulated runs of the paper (in a classical to quantum learning experiment). Photon count seems non-influential in this specific setting. |
| Adversarial Learning. Source: [lu_2020](https://arxiv.org/abs/2001.00030) | We observed on a photonic model, just like the authors of the paper on a gate-based model, that quantum classifiers are vulnerable to direct and transferred adversarial attacks but adversarial training is also effective against specific attack types. MNIST classification (1 vs 9): - Clean accuracy = 98%<br>- Adversarial accuracy (BIM, $\epsilon=0.1$) = 15%<br>- Adversarial accuracy post-adversarial training (BIM) = 95% |
| [Photonic Quantum Memristor](papers/qrc_memristor/). Source: [selimovic_2025](https://arxiv.org/abs/2504.18694) | We developed a photonic quantum reservoir computing architecture with the addition of a quantum memristor which acts as a feedback loop (memory). We noticed that in both cases the use of the memristor enhances the non-linear capabilities of the model and this leads to improved performance compared to the non-memristor case and some classical benchmarks. The results are similar to those presented in the paper and the error is 5 times smaller on the learning task of the NARMA dataset.|



## Running existing reproductions

- Browse the up-to-date catalogue at [https://merlinquantum.ai/reproduced_papers/index.html](https://merlinquantum.ai/reproduced_papers/index.html) to pick the paper you want to execute. Every paper now lives under `papers/<NAME>/`; the `<NAME>` you pass to the CLI is just that folder name (e.g., `QLSTM`, `QORC`, `reproduction_template`).

You can also list available reproductions with `python implementation.py --list-papers`.

- `cd` into `papers/<NAME>` and install its dependencies: `pip install -r requirements.txt` (each reproduction keeps its own list).
- Launch training/eval runs through the shared CLI from the repo root (the runner will `cd` into the project automatically):

	```bash
	python implementation.py --paper <NAME> --config configs/<config>.json
	```

- If you prefer running from inside `papers/<NAME>`, reference the repo-level runner: `python ../../implementation.py --config configs/<config>.json` (no `--paper` flag needed when executed from within the project).

All logs, checkpoints, and figures land in `papers/<NAME>/outdir/run_YYYYMMDD-HHMMSS/` unless the configs specify a different base path.

Need a quick tour of a project’s knobs? Run `python implementation.py --paper <NAME> --help` to print the runtime-generated CLI for that reproduction (dataset switches, figure toggles, etc.) before launching a full experiment.

### Data location

- Default data root is `data/` at the repo root; each paper writes under `data/<NAME>/` to avoid per-venv caches.
- Override with `DATA_DIR=/abs/path` or `python implementation.py --data-root /abs/path ...` (applies to the current run and is exported to downstream loaders).

Shared data helpers:
- Common dataset-generation code lives under `papers/shared/<paper>/` when multiple reproductions reuse the same logic. Each paper exposes a thin `lib/data.py` (or equivalent) that simply imports from the shared module.
- If you add new shared data utilities, place them in `papers/shared/<paper>/` and have paper-local `lib/` importers forward to them so tests and runners stay stable.

Universal CLI flags provided by the shared runner:
- `--seed INT` Reproducibility seed propagated to Python/NumPy/PyTorch backends.
- `--dtype STR` Force a global tensor dtype before model-specific overrides.
- `--device STR` Torch device string (`cpu`, `cuda:0`, `mps`, ...).
- `--log-level LEVEL` Runtime logging verbosity (`INFO` by default).
Project-specific `cli.json` files only declare the extra paper knobs; the runner injects the global options automatically.

### Smoke-test all papers quickly

- From repo root, run the portable smoke harness [scripts/smoke_test_all_papers.sh](scripts/smoke_test_all_papers.sh) to install per-paper venvs under `.smoke_envs/`, execute each paper’s default config, and run `pytest`, logging to `.smoke_logs/<paper>.log`.
- Pass an optional substring to target specific papers (faster dev loop): `scripts/smoke_test_all_papers.sh QRKD` only runs papers whose names contain `QRKD`.
- Timeout markers appear in logs when a run or test exceeds the limit; rerun after adjusting configs or deps as needed.

### Precision control (`dtype`)

- Every reproduction accepts an optional top-level `"dtype"` entry in its configs, just like `"seed"`. When present, the shared runner casts input tensors and initializes models in that dtype.
- Individual models can still override via `model.dtype`; if omitted, each reproduction picks a sensible default (e.g., `float64` for photonic MerLin layers).
- Use this to downgrade to `float32` for speed, experiment with `bfloat16`, or enforce `float64` reproducibility across classical/quantum variants.

## How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines in the [how_to_contribute](how_to_contribute.md) file
