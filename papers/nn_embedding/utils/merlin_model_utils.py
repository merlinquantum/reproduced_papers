"""
LLM helped me a lot to to implemented in details about what I wanted here
"""

import sympy as sp
import merlin as ml
import torch


def ordered_variable_params(circuit):
    seen = []
    seen_ids = set()
    for p in circuit.get_parameters(all_params=False, expressions=False):
        if p.fixed:
            continue
        if id(p) not in seen_ids:
            seen.append(p)
            seen_ids.add(id(p))
    return seen


def rename_params_in_current_order(
    circuit,
    prefix,
):
    params = ordered_variable_params(circuit)
    suffixes = [p.name for p in params]
    ordered_new_names = [f"{prefix}{name}" for name in suffixes]

    if len(params) != len(ordered_new_names):
        raise ValueError("Parameter count mismatch")

    new_params = {}
    for param, new_name in zip(params, ordered_new_names):
        param.name = new_name
        if getattr(param, "_symbol", None) is not None:
            param._symbol = sp.symbols(new_name, real=True)
        new_params[new_name] = param

    circuit._params = new_params


def strip_simple_negation_expressions(component):
    if hasattr(component, "_components"):
        for _, subcomp in component._components:
            strip_simple_negation_expressions(subcomp)

    if not hasattr(component, "_params"):
        return

    for key, param in list(component._params.items()):
        if not getattr(param, "_is_expression", False):
            continue
        base_params = list(getattr(param, "parameters", []))
        if len(base_params) != 1:
            continue
        base = base_params[0]
        if param.name == f"(-{base.name})":
            component._params[key] = base


def count_parameters_with_prefixes(circuit, prefixes):
    return sum(
        1
        for p in circuit.get_parameters(all_params=False, expressions=False)
        if any(p.name.startswith(prefix) for prefix in prefixes)
    )


def compute_x2_permutation(fidelity_layer):
    spec = fidelity_layer.computation_process.converter.spec_mappings
    x1_names = spec["x_1_"]
    x2_names = spec["x_2_"]

    x1_suffixes = [n.removeprefix("x_1_") for n in x1_names]
    x2_suffixes = [n.removeprefix("x_2_") for n in x2_names]
    return [x2_suffixes.index(s) for s in x1_suffixes]


def assign_params(layer: ml.QuantumLayer, values: torch.Tensor) -> torch.Tensor:
    if len(values.shape) == 1:
        index = 0
        for param in layer.parameters():
            new_index = index + param.numel()
            param.copy_(values[index:new_index].reshape(param.shape))
            index = new_index
        return layer()
    output = torch.empty((values.size(0), layer.output_size), dtype=layer().dtype)
    for i, value in enumerate(values):
        index = 0
        for param in layer.parameters():
            new_index = index + param.numel()
            param.copy_(value[index:new_index].reshape(param.shape))
            index = new_index
        output[i] = layer()
    return output
