# MerLin Backend Prototype Report

## Scope

This report documents how the vendored MerLin checkout in this repository differs from the pip/upstream-style `0.3.2` baseline, why those changes were made, what new capability each change enabled, and what a cleaner upstream implementation would likely look like.

The goal is not to argue that every local edit should be upstreamed exactly as written. The goal is to make the prototype legible to the MerLin team:

- what problem each change solved
- whether it was a correctness/API issue or a performance issue
- what feature became possible because of it
- what should be redesigned more cleanly upstream

Active checkout:
- [third_party/merlinquantum](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum)

Preserved pre-change backups:
- [layer.py.pre_sparse_refactor](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py.pre_sparse_refactor)
- [layer.py.pre_ebs_reroute](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py.pre_ebs_reroute)
- [process.py.pre_sparse_refactor](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py.pre_sparse_refactor)
- [probability_distribution.py.pre_sparse_refactor](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/probability_distribution.py.pre_sparse_refactor)
- [state_vector.py.pre_sparse_refactor](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/state_vector.py.pre_sparse_refactor)
- [slos_torchscript.py.pre_chunk_refactor](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/pcvl_pytorch/slos_torchscript.py.pre_chunk_refactor)

Related benchmark harness:
- [benchmark_merlin_sparse_path.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/papers/quantum_vision_transformers/scripts/benchmarks/benchmark_merlin_sparse_path.py)

## Executive Summary

The local MerLin prototype differs from the pip/upstream baseline in five important ways:

1. `StateVector` and complex amplitude inputs no longer densify immediately.
2. Superposition evaluation no longer materializes a dense `input_basis x output_basis` staging tensor.
3. `StateVector` inputs are routed onto the batched EBS path by default instead of the older incremental superposition path.
4. The TorchScript hot kernels chunk large transition lists to avoid oversized per-layer temporaries.
5. `MeasurementStrategy` can express partition-based output selection, which is lowered into final-basis filtering through `output_map_func`.

The most important practical conclusion is that the biggest win was not TorchScript micro-optimization. It was fixing the high-level dispatch and densification path.

The prototype therefore suggests that upstream MerLin most needs:

- first-class sparse amplitude handling
- default EBS dispatch for `StateVector`/superposition inputs
- first-class output filtering for structured post-selection
- removal of dense-path assumptions that tie input basis size to output basis size

## What Was Wrong in the Original Path

For the QVT workload in this repository, the original MerLin behavior had three main issues.

### 1. Sparse amplitude inputs became dense too early

The `QuantumLayer` forward path treated `StateVector` input as an amplitude-encoded state, but it normalized and densified it immediately. That meant sparse quantum states lost their sparsity before the simulation path even began.

### 2. Superposition evaluation built an unnecessary dense cross-product

Even after the input was accepted, the old process path effectively built a dense intermediate over:

- input superposition components
- output Fock states

That cost memory and time even when only a small subset of input basis states had nonzero amplitude.

### 3. `StateVector` inputs were routed to the wrong execution strategy

For QVT, `StateVector -> QuantumLayer` should have used the vectorized EBS route. Instead, the legacy dispatch logic routed the new `StateVector` path into `compute_superposition_state(...)`, which depended on incremental `compute_pa_inc(...)` updates and retained the wrong performance profile for this workload.

## High-Level Change Map

| Area | Files | What changed | What it enabled |
|---|---|---|---|
| Sparse amplitude path | `merlin/algorithms/layer.py`, `merlin/core/process.py` | preserve sparse amplitude tensors longer and normalize sparse superpositions without densifying | practical `StateVector` input for QVT-sized circuits |
| Dispatch fix | `merlin/algorithms/layer.py` | route `StateVector` / complex amplitude input to EBS | large end-to-end runtime and memory improvement |
| TorchScript chunking | `merlin/pcvl_pytorch/slos_torchscript.py` | process layer transition lists in chunks | smaller dense temporaries and moderate runtime win |
| Partition filtering | `merlin/measurement/strategies.py`, `merlin/algorithms/layer.py`, `merlin/core/process.py`, `merlin/pcvl_pytorch/slos_torchscript.py` | support partition-based `MeasurementStrategy` and lower it to final-basis filtering | structured post-selection API for QVT sector readouts |
| Reduced-basis acceptance | `merlin/core/process.py`, `merlin/algorithms/layer.py` | stop assuming amplitude-input dimension must equal full logical basis dimension | filtered/reduced superposition inputs become representable |
| Sparse-aware output objects | `merlin/core/probability_distribution.py`, `merlin/core/state_vector.py` | preserve sparse object behavior better in wrapped outputs | less accidental re-densification at the object layer |

## Detailed Changes

## 1. Sparse `StateVector` Inputs

Primary files:
- [layer.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py)
- [process.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py)

### What changed

The amplitude path in `QuantumLayer.forward(...)` now keeps `StateVector.normalize().tensor` as a tensor input without immediately calling `to_dense()`.

Representative code path:

```python
sv = input_parameters[0]
amplitude_tensor = sv.normalize().tensor
if amplitude_tensor.device != self.device:
    amplitude_tensor = amplitude_tensor.to(self.device)
if amplitude_tensor.dtype != self.complex_dtype:
    amplitude_tensor = amplitude_tensor.to(self.complex_dtype)
amplitude_input = self._validate_amplitude_input(amplitude_tensor)
```

And `_validate_amplitude_input(...)` now accepts either:

- the full logical basis size
- or a reduced filtered basis size

instead of hard-coding only `len(self.output_keys)`.

### Why

The old path assumed the dense logical-basis representation was the normal representation. That is exactly the wrong assumption for QVT, where the input superposition is often sparse and intentionally constructed as such.

### What feature this enabled

- true sparse `StateVector` input into `QuantumLayer`
- ability to carry filtered/reduced basis tensors lower into the process path
- removal of the old architectural coupling between input basis size and output basis size

### What upstream should do more cleanly

Upstream should likely formalize amplitude inputs as one of:

- dense logical basis
- sparse logical basis
- reduced filtered basis with explicit basis metadata

instead of inferring all of that only from `tensor.shape[-1]`.

## 2. Sparse-Aware Superposition Preparation

Primary file:
- [process.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py)

### What changed

The process layer now contains helpers to:

- lift reduced-basis tensors into the full logical basis when needed
- preserve sparse COO storage when unsqueezing and normalizing
- gather only active nonzero superposition coefficients

Representative helper behaviors:

```python
def _unsqueeze_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 1:
        return tensor
    if not tensor.is_sparse:
        return tensor.unsqueeze(0)
    ...
```

```python
def _normalize_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_sparse:
        ...
    coalesced = tensor.coalesce()
    ...
```

```python
def _expand_superposition_tensor(
    tensor: torch.Tensor, indices: list[int], target_dim: int
) -> torch.Tensor:
    ...
```

### Why

The old path could only handle the superposition cleanly if it was already dense and already expressed in the expected full basis. That prevented lower-level output filtering and made sparse-input workflows brittle.

### What feature this enabled

- partition-filtered or reduced-basis inputs can still flow into the process path
- sparse superposition normalization no longer forces a dense conversion
- later compute paths can stream only the active support instead of the full basis

### What upstream should do more cleanly

These basis-lifting and sparse-normalization operations should likely live in a dedicated superposition/basis utility layer rather than being spread through `ComputationProcess`.

## 3. Remove Dense `input_basis x output_basis` Staging

Primary file:
- [process.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py)

### What changed

The superposition code now streams contributions directly into the final output amplitudes instead of materializing a dense cross-product intermediate.

Representative pattern in `compute_ebs_simultaneously(...)`:

```python
final_amplitudes = torch.zeros(
    (
        parameter_batch,
        prepared_state.shape[0],
        len(self.simulation_graph.mapped_keys),
    ),
    dtype=unitary.dtype,
    device=prepared_state.device,
)

for i in range(0, len(input_states), simultaneous_processes):
    ...
    _, batch_amplitudes = self.simulation_graph.compute_batch(
        unitary, batch_fock_states
    )
    ...
    final_amplitudes += torch.einsum(
        "se, boe -> bso", coeffs, batch_amplitudes
    )
```

And `compute_superposition_state(...)` now gathers active coefficients and accumulates directly rather than allocating a large dense staging matrix.

### Why

The old memory blow-up in QVT was not only from one dense conversion. It was from:

- densify sparse input
- allocate dense per-input-component output staging
- reduce later

The staging allocation was avoidable.

### What feature this enabled

- much more practical superposition evaluation for sparse amplitude inputs
- one-photon QVT paths that no longer scale with the full inactive basis support

### What upstream should do more cleanly

The streamed accumulation pattern should become the default superposition evaluation model, with the dense cross-product path retained only when there is a measured reason to prefer it.

## 4. Dispatch `StateVector` Inputs to EBS

Primary file:
- [layer.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py)

### What changed

`_compute_amplitudes(...)` now routes:

- `self.amplitude_encoding`
- or any new vectorized amplitude input (`StateVector` / complex tensor)

to `compute_ebs_simultaneously(...)`.

Representative logic:

```python
if self.amplitude_encoding or vectorized_amplitude_input:
    ...
    return self.computation_process.compute_ebs_simultaneously(
        params, simultaneous_processes=batch_size
    )
```

### Why

For QVT, the old path was reaching `compute_superposition_state(...)` simply because the new `StateVector` input path never flipped the legacy `amplitude_encoding=True` routing flag.

This was a dispatch artifact, not a principled modeling choice.

### What feature this enabled

This was the single biggest end-to-end improvement:

- `StateVector -> QuantumLayer` benchmark
  - before reroute: about `25.52s`, peak RSS about `1.19 GB`
  - after reroute: about `0.010s`, peak RSS about `0.66 GB`

- real QVT CPU training step for `Model D`
  - before reroute: about `15.45s`, peak RSS about `2.05 GB`
  - after reroute: about `0.612s`, peak RSS about `0.81 GB`

### What upstream should do more cleanly

Upstream should not key this decision off the deprecated `amplitude_encoding=True` mode. It should dispatch based on the actual semantic input type:

- `StateVector`
- complex amplitude tensor
- classical parameter batch

In practice, `StateVector` should almost certainly default to the EBS route.

## 5. TorchScript Chunking

Primary file:
- [slos_torchscript.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/pcvl_pytorch/slos_torchscript.py)

### What changed

The hot layer kernels now chunk large transition lists.

Representative additions:

```python
_LAYER_TARGET_COMPLEX_ELEMENTS = 1 << 18

def _resolve_op_chunk_size(
    num_ops: int, batch_size: int, num_input_states: int = 1
) -> int:
    ...
```

And both `layer_compute_vectorized(...)` and `layer_compute_batch(...)` iterate over slices of:

- `sources`
- `destinations`
- `modes`

instead of materializing the whole layer transition workload in one dense block.

### Why

The precomputed graph was already sparse. The remaining issue in TorchScript was that layer execution still built large dense per-operation temporaries:

- unitary picks
- previous amplitudes
- contributions

Chunking bounded those temporaries.

### What feature this enabled

This was a useful runtime optimization, but not the dominant memory fix:

- sparse-path benchmark before chunking: about `74.16s`, peak RSS about `1.19 GB`
- after chunking: about `47.14s`, peak RSS about `1.19 GB`

So chunking helped runtime much more than RSS.

### What upstream should do more cleanly

Keep chunking, but treat it as a secondary kernel optimization. It should not distract from the higher-value dispatch and densification fixes.

## 6. Partition-Based `MeasurementStrategy`

Primary files:
- [measurement/strategies.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/measurement/strategies.py)
- [layer.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py)
- [process.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py)
- [slos_torchscript.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/pcvl_pytorch/slos_torchscript.py)

### What changed

`MeasurementStrategy` can now express partition-based output selection using:

- `partition_blocks=[...]`
- `allowed_counts=[...]`

Representative API:

```python
MeasurementStrategy.probs(
    computation_space=ComputationSpace.FOCK,
    partition_blocks=[n_patches, d],
    allowed_counts=[(1, 1)],
)
```

Strategy-side normalization and validation:

```python
blocks, counts = MeasurementStrategy._normalize_partition_selection(
    partition_blocks, allowed_counts
)
```

```python
def build_output_map_func(
    self, *, n_modes: int, n_photons: int
) -> Callable[[tuple[int, ...]], tuple[int, ...] | None] | None:
    ...
```

Layer-side lowering:

```python
if measurement_strategy.has_partition_selection():
    ...
    output_map_func = measurement_strategy.build_output_map_func(
        n_modes=circuit.m,
        n_photons=resolved_n_photons,
    )
```

And `ComputationProcess` now actually passes `output_map_func` into:

```python
self.simulation_graph = build_slos_distribution_computegraph(
    ...,
    output_map_func=self.output_map_func,
)
```

### Why

QVT does not just need generic “probabilities.” It often needs:

- cross-sector outputs only
- unions of selected sectors
- hierarchical sector patterns

The old post-selection was happening too late and too manually.

### What feature this enabled

This made the QVT readout semantics expressible in MerLin itself.

Examples:

- `D cross_only`
  - `partition_blocks=[n_patches, d]`
  - `allowed_counts=[(1, 1)]`

- `D_full`
  - `partition_blocks=[n_patches, d]`
  - `allowed_counts=[(1, 1), (2, 0), (0, 2)]`

- `F`
  - `partition_blocks=[n_regions, n_patches_per_region, d]`
  - `allowed_counts=[(1, 1, 1)]` or additional sector unions

### Important limitation

This is final-basis pruning, not full throughout-the-graph pruning.

It avoids returning disallowed final states, but it does not yet guarantee that all earlier branches leading only to disallowed sectors are eliminated during propagation.

### What upstream should do more cleanly

This should become a first-class public concept, probably as:

- partition-aware output filtering or
- structured post-selection

separate from the rest of measurement semantics.

The current prototype works, but upstream should likely separate:

- measurement kind
- output filtering
- grouping / aggregation

instead of letting all three accumulate in one strategy object.

## 7. Reduced-Basis Acceptance

Primary files:
- [layer.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/algorithms/layer.py)
- [process.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/process.py)

### What changed

The amplitude path no longer assumes that the amplitude tensor dimension must equal the final output basis size. Instead, it accepts:

- the full logical basis size
- or the reduced filtered output basis size

and lifts reduced-basis tensors lower in the process layer where necessary.

### Why

That old assumption was an artifact of the dense path. It became a blocker as soon as output filtering and reduced sparse inputs were introduced.

### What feature this enabled

- filtered superposition inputs are valid
- partition-pruned layers can accept reduced-basis tensors
- basis coupling between input and output is no longer hard-coded into the top-level layer validation

### What upstream should do more cleanly

Upstream should likely carry explicit basis metadata with amplitude objects instead of inferring reduced-basis semantics purely by dimensionality.

## 8. Sparse-Aware Output Objects

Primary files:
- [probability_distribution.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/probability_distribution.py)
- [state_vector.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum/merlin/core/state_vector.py)

### What changed

The wrapped output objects now better tolerate sparse tensors and filtered bases instead of assuming dense tensors everywhere.

In `ProbabilityDistribution`, for example, a `FilteredBasis` abstraction now exists to represent subset views over a larger basis.

### Why

Without this, it is too easy to optimize the compute path and then accidentally re-densify or lose basis meaning when wrapping the result object.

### What feature this enabled

- sparse-aware return objects for filtered distributions
- more coherent behavior when final output filtering is used

### What upstream should do more cleanly

If filtered bases are to remain supported, they should probably become a more visible part of the public object model rather than an implementation detail.

## What Each Change Made Possible

| Change | New capability |
|---|---|
| sparse `StateVector` acceptance | QVT amplitude encoding can be expressed directly without dense conversion |
| streamed superposition accumulation | practical multi-component superpositions with large logical basis but small support |
| EBS reroute | `StateVector`-driven models become fast enough to train and benchmark |
| TorchScript chunking | large transition layers become more runtime-stable |
| partition-based `MeasurementStrategy` | QVT readout/post-selection semantics become declarative and backend-visible |
| reduced-basis acceptance | filtered output spaces can feed later stages |
| sparse-aware output objects | the optimized path remains coherent after wrapping outputs |

## Prototype Benchmarks

These are the most important measured outcomes from the prototype work.

### Sparse-path benchmark

Using the local benchmark harness:
- [benchmark_merlin_sparse_path.py](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/papers/quantum_vision_transformers/scripts/benchmarks/benchmark_merlin_sparse_path.py)

Observed progression:

- before sparse-preserving pass
  - `20 modes, 3 photons, 12 terms`: about `303.98s`, peak RSS about `1,206,392 KB`
- after sparse-preserving pass
  - `20 modes, 3 photons, 12 terms`: about `74.16s`, peak RSS about `1,189,320 KB`
- after TorchScript chunking
  - `20 modes, 3 photons, 12 terms`: about `47.14s`, peak RSS about `1,192,416 KB`
- after EBS reroute
  - `20 modes, 3 photons, 12 terms`: about `0.010s`, peak RSS about `657,936 KB`

Interpretation:

- sparse preservation mattered
- chunking helped
- dispatch to EBS was the decisive win

### End-to-end QVT training step

Measured on CPU for the real QVT training path:

- `Model D`
  - before EBS reroute: about `15.45s`, peak RSS about `2,046,900 KB`
  - after EBS reroute: about `0.612s`, peak RSS about `806,380 KB`

This confirms that the EBS dispatch fix was not only a microbenchmark artifact.

## What Is QVT-Specific vs Generally Useful Upstream

### Likely generally useful upstream

- sparse amplitude handling
- EBS-by-default dispatch for `StateVector`
- chunked TorchScript layer kernels
- support for output filtering through `output_map_func`
- reduced-basis acceptance in the process layer

### More QVT-specific or still prototype-quality

- exact partition-block / allowed-count API shape
- some of the basis-lifting heuristics
- the specific choice to encode post-selection in `MeasurementStrategy` instead of a separate output-filter abstraction

## Recommended Upstream Direction

If the MerLin team wants to carry these ideas forward cleanly, the prototype suggests this direction:

1. Introduce a first-class sparse amplitude/superposition path.
   - Avoid treating dense tensors as the only “real” representation.

2. Make `StateVector` dispatch semantic, not legacy-flag-based.
   - `StateVector` should use EBS/vectorized execution by default.

3. Split measurement semantics from output filtering.
   - Keep `probabilities` / `amplitudes` / `partial`
   - separate from “filter the final output basis to this structured sector set”

4. Formalize reduced-basis metadata.
   - Do not infer everything from `shape[-1]`.

5. Keep chunking as a kernel-level optimization, but treat it as secondary.
   - The real design problem was above the TorchScript layer.

6. Clarify physical vs nonphysical restrictions.
   - true post-selection / output filtering
   - versus throughout-the-graph restrictions like `UNBUNCHED`

## Suggested Upstream Cleanup Tasks

If upstream were to adopt this work, the likely cleanup plan would be:

1. Re-implement the sparse/eager basis utilities in a dedicated module.
2. Add formal tests for:
   - sparse `StateVector` input
   - EBS dispatch
   - filtered-basis input and output
   - partition-based output filtering
3. Decide whether partition filtering belongs in:
   - `MeasurementStrategy`
   - a separate `OutputFilter`
   - or graph-construction options
4. Reduce heuristic coupling in `ComputationProcess`.
5. Upstream chunking in TorchScript with tunable or adaptive policy.

## Bottom Line

This prototype shows that the main MerLin issue for the QVT workload was not “TorchScript is too slow” in isolation. It was:

- dense-path assumptions
- wrong superposition dispatch
- lack of backend-visible structured output filtering

The prototype fixes made three important things possible:

- practical `StateVector` training for QVT
- structured partition-based post-selection inside MerLin
- reduced-basis workflows that are not immediately rejected by dense assumptions

So the most useful message to the MerLin team is:

> the prototype should be read less as a set of patches and more as evidence that MerLin needs a first-class sparse superposition path, semantic `StateVector` dispatch, and explicit output-filter support.
