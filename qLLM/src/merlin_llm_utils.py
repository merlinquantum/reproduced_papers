"""
MerLin models for the qLLM with different setups.
"""

import math

import merlin as ml  # Using our Merlin framework
import perceval as pcvl
import torch
import torch.nn as nn

##########################################
### Quantum classifier and its wrapper ###
##########################################


def create_quantum_circuit(m, size=400):
    """Create a quantum circuit with specified number of modes and input size"""

    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_1_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_2_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    c_var = pcvl.Circuit(m)
    for i in range(size):
        px = pcvl.P(f"px-{i + 1}")
        c_var.add(i % m, pcvl.PS(px))
    print(c_var)
    c.add(0, c_var, merge=True)

    wr = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_3_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_4_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c.add(0, wr, merge=True)

    return c


def create_quantum_circuit_deep(m, size=400):
    """Create a quantum circuit with specified number of modes and input size
    The encoding is done in single rows of phase shifters then trainable interferometers
    """

    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_1_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_2_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)
    nb_encoding = size // m + 1
    for layer_idx in range(nb_encoding):
        c_var = pcvl.Circuit(m)
        idx_max = min(m * layer_idx + m, size)
        for i in range(layer_idx * m, idx_max):
            px = pcvl.P(f"px-{i + 1}")
            c_var.add(i % m, pcvl.PS(px))
        print(c_var)
        c.add(0, c_var, merge=True)

        wr = pcvl.GenericInterferometer(
            m,
            lambda i, id=layer_idx: pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_3_{id}_{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_4_{id}_{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        c.add(0, wr, merge=True)

    return c


class ScaleLayer(nn.Module):
    def __init__(self, dim, scale_type="learned"):
        super().__init__()
        # Create a single learnable parameter (initialized to 1.0 by default)
        # Caution: MerLin already mutltiplies by pi
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)
        # print(f"SELF.SCALE: {self.scale.shape}")

    def forward(self, x):
        # Element-wise multiplication of each input element by the learned scale
        return x * self.scale


class DivideByPi(nn.Module):
    """Normalizes phases to be within quantum-friendly range"""

    def forward(self, x):
        return x / torch.pi


# First, we propose a very basic version of a QuantumClassifier following a simple architecture from Gan et al.


class QuantumClassifier(nn.Module):
    """
    This QuantumClassifier consists of a simple "Sandwich" architecture to encode the data :
        - data encoding:
            - the 768 embeddings are mapped to a smaller space using a Linear Layer
            - the data is scaled by a learnable layer (= size of the data) and scaled between 0 an 1;
            - the data points are encoded in phase shifters
        - quantum layer : [trainable interferometer] - data encoding - [trainable interferometer]
        - outputs: the output of the quantum layer are mapped to num_classes using a Linear Layer
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )
        # this layer downscale the inputs to fit in the QLayer
        print(f"Input dim = {input_dim} and hidden dim = {hidden_dim}")
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.scale = ScaleLayer(dim=hidden_dim, scale_type="learned")
        self.sig = nn.Sigmoid()
        self.pi_scale = DivideByPi()

        # building the QLayer with MerLin
        circuit = create_quantum_circuit(modes, size=hidden_dim)
        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)
        # output_size of the interferometer
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )
        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        print("\n -> self.q_circuit")
        self.q_circuit = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit,
            trainable_parameters=[
                p.name for p in circuit.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )
        self.bn = nn.LayerNorm(output_size_slos).requires_grad_(False)  # works OK
        print(f"\n -- Building the Linear layer with output size = {num_classes} -- ")
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        out = self.downscaling_layer(x)
        # casts the input to [0,1] + use q_circuit
        out = self.sig(out)
        # division by pi
        # out = self.pi_scale(self.sig(out))

        out = self.q_circuit(out)  # self.sig
        out = self.output_layer(out)
        return out


# Second, we propose a more advanced version using 2 quantum modules (1 that can be parallelised and one that process the quantum outputs)


class QuantumClassifierParallel(nn.Module):
    """
    This QuantumClassifier consists of 2 encoders:
        - a first module that can parallelize E = 1 or E = 2 circuits and uses angle encoding (sandwich - not bunched)
        - a second module that process the outputs of the first module in a similar manner
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
        e=1,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )

        # this layer downscale the inputs to fit in the QLayer
        hidden_dim = modes
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.pi_scale = DivideByPi()
        self.sig = nn.Sigmoid()
        self.E = e

        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)

        ### FIRST MODULE ###
        # building the 2 parallel QLayer with MerLin and non bunched modes
        circuit_1 = create_quantum_circuit(modes, size=hidden_dim)
        circuit_2 = create_quantum_circuit(modes, size=hidden_dim)

        self.q_circuit_1 = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit_1,
            trainable_parameters=[
                p.name
                for p in circuit_1.get_parameters()
                if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=True,
        )
        if self.E == 2:
            self.q_circuit_2 = ml.QuantumLayer(
                input_size=hidden_dim,
                output_size=None,  # but we do not use it
                circuit=circuit_2,
                trainable_parameters=[
                    p.name
                    for p in circuit_2.get_parameters()
                    if not p.name.startswith("px")
                ],
                input_parameters=["px"],
                input_state=input_state,
                output_mapping_strategy=ml.OutputMappingStrategy.NONE,
                device=device,
                no_bunching=True,
            )

        output_encoder = math.comb(modes, photons_count)

        ### SECOND MODULE ###
        # PNR output size
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )

        print("\n -> self.q_circuit")

        circuit_final = create_quantum_circuit(modes, size=self.E * output_encoder)

        self.q_circuit_final = ml.QuantumLayer(
            input_size=self.E * output_encoder,
            output_size=None,  # but we do not use it
            circuit=circuit_final,
            trainable_parameters=[
                p.name
                for p in circuit_final.get_parameters()
                if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )

        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        # forward pass
        x = self.downscaling_layer(x)
        x = self.sig(x)
        x = self.pi_scale(x)

        ### first module ###
        # first encoder
        e1 = self.q_circuit_1(x)
        # no concatenation needed
        e_cat = e1
        if self.E == 2:
            # second encoder if necessary
            e2 = self.q_circuit_2(x)
            # concatenation of the 2 outputs
            e_cat = torch.cat([e1, e2], dim=1)
        # casts the input to [0,1] + use q_circuit
        out = self.sig(e_cat)

        ### second module ###
        out = self.q_circuit_final(out)

        ### final classification layer ###
        out = self.output_layer(out)

        return out


# Then, we propose another advanced version using:
# - 2 quantum modules with deep circuits (1 that can be parallelised and one that process the quantum outputs);
# - a measurement based on the expectation value of getting at least 1 photon in each mode


def marginalize_photon_presence(keys, probs):
    """
    Marginalize Fock state probabilities to get per-mode occupation probabilities.

    Computes the probability that each mode contains at least one photon
    by summing over all Fock states where that mode is occupied.

    Args:
        keys (list): List of Fock state tuples, e.g., [(0,1,0,2), (1,0,1,0), ...]
        probs (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
            for each Fock state, with requires_grad=True

    Returns:
        torch.Tensor: Shape (N, num_modes) with marginal probability that
            each mode has at least one photon
    """
    device = probs.device
    keys_tensor = torch.tensor(
        keys, dtype=torch.long, device=device
    )  # shape: (num_keys, num_modes)
    keys_tensor.shape[1]

    # Create mask of shape (num_modes, num_keys)
    # Each mask[i] is a binary vector indicating which Fock states have >=1 photon in mode i
    mask = (keys_tensor >= 1).T  # shape: (num_modes, num_keys)

    # Convert to float to allow matrix multiplication
    mask = mask.float()

    # Now do: (N, num_keys) @ (num_keys, num_modes) → (N, num_modes)
    marginalized = probs @ mask.T  # shape: (N, num_modes)
    return marginalized


class QuantumClassifierExpectation(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
        e=1,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )
        # this layer downscale the inputs to fit in the QLayer
        hidden_dim = modes
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.pi_scale = DivideByPi()
        self.sig = nn.Sigmoid()
        self.E = e

        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)

        ### FIRST MODULE ###

        # building the QLayer with MerLin
        circuit_1 = create_quantum_circuit_deep(modes, size=hidden_dim)
        circuit_2 = create_quantum_circuit_deep(modes, size=hidden_dim)

        self.q_circuit_1 = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit_1,
            trainable_parameters=[
                p.name
                for p in circuit_1.get_parameters()
                if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=True,
        )
        if self.E == 2:
            self.q_circuit_2 = ml.QuantumLayer(
                input_size=hidden_dim,
                output_size=None,  # but we do not use it
                circuit=circuit_2,
                trainable_parameters=[
                    p.name
                    for p in circuit_2.get_parameters()
                    if not p.name.startswith("px")
                ],
                input_parameters=["px"],
                input_state=input_state,
                output_mapping_strategy=ml.OutputMappingStrategy.NONE,
                device=device,
                no_bunching=True,
            )

        # We generate the keys associated with the probs
        dummy_input = torch.zeros(1, hidden_dim, dtype=torch.float32)
        # Get MerLin probability distribution (no gradients to avoid sampling warnings)
        with torch.no_grad():
            merlin_params = self.q_circuit_1.prepare_parameters([dummy_input])
            unitary = self.q_circuit_1.computation_process.converter.to_tensor(
                *merlin_params
            )
            # perfect distribution (no sampling)
            self.keys, probs = (
                self.q_circuit_1.computation_process.simulation_graph.compute(
                    unitary, input_state
                )
            )

            probs_marginalized = marginalize_photon_presence(self.keys, probs)

            print(f"Probabilities marginalized with shape {probs_marginalized.shape}")
            output_encoder = probs_marginalized.shape[-1]
            print(f"Therefore the input of the encoder is of shape {output_encoder}")

        ### SECOND MODULE ###
        circuit_final = create_quantum_circuit(modes, size=self.E * output_encoder)

        self.q_circuit_final = ml.QuantumLayer(
            input_size=self.E * output_encoder,
            output_size=None,  # but we do not use it
            circuit=circuit_final,
            trainable_parameters=[
                p.name
                for p in circuit_final.get_parameters()
                if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )

        # PNR output size
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )

        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        # forward pass
        x = self.downscaling_layer(x)
        out = self.sig(x)
        x = self.pi_scale(x)

        ### first module ###
        e1 = self.q_circuit_1(x)
        e1_expectation = marginalize_photon_presence(self.keys, e1)
        e_cat = e1_expectation
        if self.E == 2:
            e2 = self.q_circuit_2(x)
            e2_expectation = marginalize_photon_presence(self.keys, e2)
            e_cat = torch.cat([e1_expectation, e2_expectation], dim=1)
        # casts the input to [0,1] + use q_circuit
        out = self.sig(e_cat)

        ### second module ###
        out = self.q_circuit_final(out)

        ### final classification layer ###
        out = self.output_layer(out)
        return out


def test_module_building_and_gradients():
    """
    Test function to verify module building and gradient propagation for all quantum classifiers.

    Tests:
    1. Module instantiation
    2. Forward pass
    3. Gradient computation and propagation
    4. Parameter updates
    """
    print("=" * 60)
    print("Testing Module Building and Gradient Propagation")
    print("=" * 60)

    # Test parameters
    input_dim = 768
    hidden_dim = 10
    modes = 10
    num_classes = 2
    batch_size = 4
    device = "cpu"

    # Create dummy data
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    y = torch.randint(0, num_classes, (batch_size,))

    # Test configurations
    test_configs = [
        ("QuantumClassifier", QuantumClassifier),
        (
            "QuantumClassifierParallel (E=1)",
            lambda **kwargs: QuantumClassifierParallel(**kwargs, E=1),
        ),
        (
            "QuantumClassifierParallel (E=2)",
            lambda **kwargs: QuantumClassifierParallel(**kwargs, E=2),
        ),
        (
            "QuantumClassifier_expectation (E=1)",
            lambda **kwargs: QuantumClassifierExpectation(**kwargs, E=1),
        ),
        (
            "QuantumClassifier_expectation (E=2)",
            lambda **kwargs: QuantumClassifierExpectation(**kwargs, E=2),
        ),
    ]

    results = {}

    for name, model_class in test_configs:
        print(f"\n{'=' * 40}")
        print(f"Testing {name}")
        print(f"{'=' * 40}")

        try:
            # Test 1: Module instantiation
            print("1. Testing module instantiation...")
            model = model_class(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                modes=modes,
                num_classes=num_classes,
                device=device,
                no_bunching=False,
            )
            print(f"   ✓ {name} instantiated successfully")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"   - Total parameters: {total_params}")
            print(f"   - Trainable parameters: {trainable_params}")

            # Test 2: Forward pass
            print("2. Testing forward pass...")
            model.eval()
            with torch.no_grad():
                output = model(x)
            print("   ✓ Forward pass successful")
            print(f"   - Input shape: {x.shape}")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")

            # Test 3: Gradient computation
            print("3. Testing gradient computation...")
            model.train()

            # Forward pass with gradients
            output = model(x)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, y)
            print(f"   - Loss: {loss.item():.6f}")

            # Backward pass
            loss.backward()
            print("   ✓ Backward pass successful")

            # Test 4: Check gradients
            print("4. Checking gradient propagation...")
            grad_stats = {}
            for name_param, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_stats[name_param] = grad_norm
                        if grad_norm > 1e-8:
                            print(f"   ✓ {name_param}: grad_norm = {grad_norm:.6e}")
                        else:
                            print(
                                f"   ⚠ {name_param}: grad_norm = {grad_norm:.6e} (very small)"
                            )
                    else:
                        print(f"   ✗ {name_param}: No gradient computed")

            # Check for vanishing/exploding gradients
            if grad_stats:
                max_grad = max(grad_stats.values())
                min_grad = min(grad_stats.values())
                avg_grad = sum(grad_stats.values()) / len(grad_stats)

                print("   - Gradient statistics:")
                print(f"     Max: {max_grad:.6e}")
                print(f"     Min: {min_grad:.6e}")
                print(f"     Avg: {avg_grad:.6e}")

                if max_grad > 10:
                    print(
                        f"   ⚠ Warning: Large gradients detected (max: {max_grad:.2e})"
                    )
                elif max_grad < 1e-6:
                    print(
                        f"   ⚠ Warning: Very small gradients detected (max: {max_grad:.2e})"
                    )
                else:
                    print("   ✓ Gradients appear normal")

            # Test 5: Parameter update simulation
            print("5. Testing parameter updates...")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Store initial parameters
            initial_params = {}
            for name_param, param in model.named_parameters():
                if param.requires_grad:
                    initial_params[name_param] = param.data.clone()

            # Take optimization step
            optimizer.step()

            # Check parameter updates
            updated = 0
            for name_param, param in model.named_parameters():
                if param.requires_grad and name_param in initial_params:
                    param_change = (
                        (param.data - initial_params[name_param]).norm().item()
                    )
                    if param_change > 1e-8:
                        updated += 1

            print(f"   ✓ {updated}/{len(initial_params)} parameters updated")

            results[name] = {
                "status": "SUCCESS",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "output_shape": output.shape,
                "loss": loss.item(),
                "grad_stats": grad_stats,
                "params_updated": updated,
            }

            print(f"   ✓ All tests passed for {name}")

        except Exception as e:
            print(f"   ✗ Error in {name}: {str(e)}")
            results[name] = {"status": "FAILED", "error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    successful_tests = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    total_tests = len(results)

    print(f"Successful tests: {successful_tests}/{total_tests}")

    for name, result in results.items():
        if result["status"] == "SUCCESS":
            print(
                f"✓ {name}: {result['trainable_params']} trainable params, loss={result['loss']:.4f}"
            )
        else:
            print(f"✗ {name}: {result['error']}")

    if successful_tests == total_tests:
        print("\n🎉 All module building and gradient tests passed!")
    else:
        print(f"\n⚠ {total_tests - successful_tests} tests failed.")

    return results


if __name__ == "__main__":
    test_module_building_and_gradients()
