# Limitations of Amplitude Encoding on Quantum Classification

## Reference and Attribution

- Paper: Limitations of Amplitude Encoding on Quantum Classification(2025)
- Authors: Wang *et al.*
- DOI/ArXiv: [2503.01545](https://arxiv.org/abs/2503.01545)


## Overview

### 🎯 Main goal 
> Explore the limitations of amplitude encoding whileseeing if the same results appear with angle encoding.

### Main result

> “We have investigated the concentration phenomenon induced by amplitude encoding and the resulting loss barrier phenomenon. Since encoding is a necessary and crucial step in leveraging QML to address classical problems, our findings indicate that the direct use of amplitude encoding may undermine the potential advantages of QML. Therefore, more effort should be devoted to developing more efficient encoding strategies to fully unlock the potential of QML.”

A loss function barrier seems to appear with angle encoding for most of the complex datasets. If the features are not espacially sparse, the loss will plateau. 

Talking about the concentration phenomena:

> “However, our numerical simulations reveal that under amplitude encoding as the amount of training data increases, although the generalization error decreases, the training error increases counterintuitively, resulting in overall poor prediction performance.”

### Main theorems and propositions of the paper
> **Theorem**:  For a K-class classification, we employ the cross-entropy loss function $LS(\theta)$ [...]. The quantum classifier is trained on a balanced training set $S = \{(x (m) , y(m) )\}^M_{ m=1}$, where each class contains M/K samples. Suppose the eigenvalues of each observable $H_k$ belong to [-1, 1], for k = 1, ..., K. If the trace distance between the expectations of encoded states of any different classes is less than $\epsilon$, then for any PQC U($\theta$) and optimization algorithm, we have $LS(\theta) \geq \ln [K - 4(K - 1)\epsilon]$ with probability at least $1 - 8e^{-M\epsilon^2/8K}$.

> **Proposition**: Assume that all elements in the feature $x \in \mathbb{R}^{2^n}$ have the same sign, and the elements satisfy $|x_i| \in [m, M]$. If $|\frac{m}{M}-1|<\epsilon$, then after amplitude encoding, we have
$$ T\bigg(\rho(x), \frac{1}{2^n}\ket{+}^{\otimes n}\bra{+}^{\otimes n}\bigg)$$
where $T(\rho_1,\rho_2)=\frac{1}{2}||\rho_1-\rho_2||_1$, the Schatten-1 norm.

Here we see that no mater the distribution, the encoded state, no matter the associated class will converge towards rge same state.

> The other two proposition are more refined that the previous one. The first one says that if a feature as a symetric density function and a mean value of zero, the encoded state will converge to the completly mixed state. The other one says that if the density function of a feature for two classes for a point x is oppsoite sign-wise, the expected encoded state will be the same for both classes

### Their framework
1. The data is encoding via amplitude encoding
2. The trainable layer:
  a. For one qubit, one layer is composed of RZ, RX, RZ. We use L of them
  b. For multiple qubits (here 10): a gate-based qcnn is used.

  ![](images/qiskit_qcnn.png)
  Source: X. Wang, Y. Wang, B. Qi, and R. Wu, “Limitations of Amplitude Encoding on Quantum Classification,” Mar. 03, 2025, arXiv: arXiv:2503.01545. doi: [10.48550/arXiv.2503.01545](https://arxiv.org/abs/2503.01545).


### Difference in framework
In a MerLin point of view, we can not directly adopt gate based circuits to photnics. Instead we inspired ourselves from their models to create our own.

1. The data is encoded via amplitude encoding.
2. The trainable layer:
   1. For one qubit, one layer is composed of random phase shifters. We use `L` of them.
   2. For multiple qubits:
      1. If it's a simple model, we add `L` layers of `CircuitBuilder.add_entangling_layer()`.
      2. If it is a QCNN model, we use the hybrid model presented in the [photonic_QCNN folder](../photonic_QCNN/).



For simple layers, the angle amplitud emodel was also implemented.
1. A ``CircuitBuilder.add_entangling_layer()`` or, if their is one single mode a ``CircuitBuilder.add_rotations()`` is added at the start of the circuit.
2. The data is encoding via amplitude encoding
3. We add L-1 layers of ``CircuitBuilder.add_entangling_layer()`` or ``CircuitBuilder.add_rotations()`` depending on the number of modes.

### Their results
TODO
### Our results
TODO
## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the paper-level `lib/runner.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/reproduction_template
python l../../implementation.py  --help

# From the repo root
python implementation.py --paper DQNN --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config CONFIG_NAME` Load an additional JSON config (merged over `defaults.json`). The config path is automatically handled by the code.

Example runs:

```bash
# From a JSON config (inside the project)
python ../../implementation.py  --config configs/defaults.json

# Override some parameters inline
python ../../implementation.py  --config configs/defaults.json --TODO 50 

# Equivalent from the repo root
python implementation.py --paper AA_study --config configs/defaults.json --TODO 50
```

## Project structure --> TODO
- `papers.DQNN.lib/runner.py` — The file to run for every experiment.
- `papers.DQNN.lib/` — core papers.DQNN.library modules used by scripts.
  - `torchmps/` — Repository to instanciate a MPS tensor module in Torch.
  - `ablation_exp.py`, `bond_dimension_exp.py`, `default_exp.py`- Files containing the function to run the corresponding experiment.
  - `boson_sampler.py` - The file containg the class managing the quantum layers.
  - `classical_utils.py`, `photonic_qt_utils.py` - Files containing utility functions.
  - `model.py` — The torch module implementing the quantum train algorithm.
- `configs/` — Experiment configs + CLI schema consumed by the shared runner. The available ones are below.
  - `defaults.json`, `cli.json`, `bond_dim_exp.json`, `ablation_exp.json`
- Other
  - `requirements.txt` — Python dependencies.
  - `tests/` - Unitary tests to make sure the papers.DQNN.library works correctly.
  - `utils/` — Containing the `utils.py` file used for plotting and repo utility functions.

## Results and Analysis

- The results are stored in the [results](results/) folder. Logs and figures will be saved in the [outdir](outdir/) directory.
- To reproduce the experiments, simply call these lines at the paper level:
 
 For just a basic training and evaluation:
 >``python3 ../../implementation.py  --config defaults.json``



## Extensions and Next Steps


## Testing

Run tests from inside the `papers/AA_study/` directory:

```bash
cd papers/AA_study
pytest -q
```
Notes:
- Tests are scoped to this template folder and expect the current working directory to be `DQNN/`.
- If `pytest` is not installed: `pip install pytest`.

## Acknowledgments