# Attribution and Distribution Scope

The conditional GAN, tokenizers, dataset interface, and Gaussian reference are
adapted from [ORCA Computing's PEP-Q-GAN implementation](https://github.com/orcacomputing/pep-q-gan),
under the [Apache-2.0 license](LICENSE.pep-q-gan). Imported and distributed
model source hashes are recorded in [assets/upstream_sources.json](assets/upstream_sources.json).

Local changes include package-relative imports, batch-safe squeeze behavior,
 checkpointable streams, and shared-runtime integration. The selected extension uses a fixed MerLin
entangling-layer circuit and 16-photon Fock input. It is a simulator adaptation
of the paper, not an identical squeezed-state GBS or hardware reproduction.

This release targets Dist/Indist and a Gaussian reference only. MerLin constructs
the circuit; Perceval supplies official tagged Dist and full-Fock Indist event
sampling. Condition-specific analytic moments standardize the stored occupations.
GAN training uses PyTorch.

Public files comprise implementation, configuration, documentation, public circuit
recipes, and compact selected-experiment evidence. Datasets, external model
weights, raw pools, checkpoints, privately supplied optical assets, internal
backend code, and server records are not bundled.

Dataset rights are separate from this code license. The author training-file
URLs/hashes are recorded for reproducibility. IEDB reconstruction and HLA
pseudosequence conditions are documented in [DATA_SOURCES.md](DATA_SOURCES.md).
MHCflurry is installed and its models downloaded by the user; no installation or weights are redistributed.