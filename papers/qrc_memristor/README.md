# Quantum Reservoir Computing with (Quantum) Memristor

This repository contains the reproduction code for **Quantum Reservoir Computing (QRC)** enhanced with memristors, alongside classical baseline models for comparison.

## Reference and Attribution

- **Paper**: *Experimental neuromorphic computing based on quantum memristor (2025)*
- **Authors**: *Mirela Selimovic, Iris Agresti, Michal Siemaszko, Joshua Morris, Borivoje Dakic, Riccardo Albiero, Andrea Crespi, Francesco Ceccarelli, Roberto Osellame, Magdalena Stobinska, and Philip Walther*
- **DOI/ArXiv**: *https://arxiv.org/pdf/2504.18694*

## Overview

The goal of this project is to reproduce and validate the performance of a hybrid quantum-memristive reservoir computing model. The project compares this architecture against standard classical benchmarks on complex time-series forecasting and non-linear transformation tasks.

### Scope
- **Models**:
  - **Quantum**: Quantum Reservoir with Memristor (`memristor`), Quantum Reservoir without Memristor (`nomem`).
  - **Classical**: Linear (`L`), Quadratic (`Q`), Linear with Memory (`L+M`), Quadratic with Memory (`Q+M`).
- **Tasks (for now only NARMA10 and nonlinear function transormation work as epxected)**:
  - NARMA10 (Time-series prediction)
  - Non-linear function transformation
  - Mackey-Glass
  - Santa Fe Laser Data
- **Implementation**: PyTorch, NumPy, Scikit-Learn.

## How to Run

### Install dependencies

It is recommended to use a virtual environment to manage dependencies.
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Command-Line Interface (CLI)

The main entry point is the repository-level `implementation.py`. You must target this specific paper folder using the 
`--paper` argument.

```
python implementation.py --paper qrc_memristor [ARGUMENTS]
```

### Running Quantum Experiments

By default, the runner executes the Quantum models (`run_quantum.py`).

**Example: Run NARMA task with Memristor**
```
python implementation.py --paper qrc_memristor --task narma --model-type memristor --memory 4 --n-runs 10
```

**Example: Run Non-linear task without Memristor**
```
python implementation.py --paper qrc_memristor --task nonlinear --model-type nomem --epochs 200
```

### Running Classical Benchmarks

To run the classical baselines, add the `--mode classical` flag.

**Example: Run Linear Baseline (L)**
```
python implementation.py --paper qrc_memristor --mode classical --task narma --model-type L
```

**Example: Run Quadratic Model with Memory (Q+M)**
```
python implementation.py --paper qrc_memristor --mode classical --task narma --model-type Q+M --epochs 500
```

### Using Configuration Files

You can load parameters from JSON files located in `reproduced_papers/qrc_memristor/configs/`. CLI arguments will override 
values in the config file.
```
# Load specific config for Nonlinear Memristor
python implementation.py --paper qrc_memristor --config configs/nonlinear_memristor.json

# Load config but override the learning rate
python implementation.py --paper qrc_memristor --config configs/nonlinear_memristor.json --lr 0.001
```

#### Configuration & Arguments

The CLI accepts the following arguments (see `configs/cli.json` for details):

| Argument | Description                                        | Options / Examples |
| :--- |:---------------------------------------------------| :--- |
| `--mode` | Select experiment script                           | `quantum` (default), `classical` |
| `--task` | Task to run (narma and nonlinear work as expected) | `narma`, `nonlinear`, `mackey_glass`, `santa_fe` |
| `--model-type` | Model architecture                                 | **Q**: `memristor`, `nomem`<br>**C**: `L`, `Q`, `L+M`, `Q+M` |
| `--memory` | Memory depth                                       | `4`, `10`, `20` (Integer) |
| `--n-runs` | Number of independent runs                         | `10` (Integer) |
| `--epochs` | Training epochs                                    | `100`, `200` (Integer) |
| `--lr` | Learning rate                                      | `0.01`, `0.05` (Float) |
| `--output-dir` | Output directory base                              | `results/`, `results_classical/` |


## Results and Analysis

At each run, a timestamped folder is created (e.g., `results/narma_mem10_20260128_...`).

**Output Files:**
- `config.json`: The exact configuration used for the run. 
- `metrics.json`: Final performance metrics (MSE, Standard Deviation). 
- `plot_data.json`: Raw prediction arrays for plotting. 
- `experiment.log`: Execution logs.

**Generating Plots:**

If you run with the `--plot` flag, plots are generated automatically. They are placed in the same folder as the other 
results of the experiment. To generate them manually from a saved run:
```
python reproduced_papers/qrc_memristor/utils/create_plots.py --result_dir results/your_run_folder_name
```

You can also generate plots similar to those the paper contains in the Jupyter Notebook of the repository 
(`notebook.ipynb`). The plots that are generated in the notebook contain more details compared to the ones that are 
created when the models are running. 

For example, this is the Figure that can be generated from the notebook and 
reproduces Figure 3 of the paper.

![alt text](utils/images/figure3.png)





