# import unittest
# import torch
# import gymnasium as gym
# import numpy as np

# # Importation des modules depuis l'implémentation fournie
# from lib.util import (
#     MinigridImageOnlyWrapper,
#     HybridMLPModel,
#     rl_agent_forward,
#     compute_discounted_returns
# )

# class TestQTRLSanity(unittest.TestCase):
#     def test_minigrid_wrapper(self):
#         '''Test de l'environnement et du traitement des données (Data Loading)'''
#         try:
#             base_env = gym.make("MiniGrid-Empty-5x5-v0")
#             env = MinigridImageOnlyWrapper(base_env)
#             obs, _ = env.reset()

#             # Vérification de la planéité et du type des données
#             self.assertIsInstance(obs, np.ndarray)
#             self.assertEqual(len(obs.shape), 1)  # Doit être 1D
#             env.close()
#         except gym.error.NameNotFound:
#             self.skipTest("Gymnasium MiniGrid n'est pas installé, test ignoré.")

#     def test_hybrid_mlp_model_forward(self):
#         '''Test de l'initialisation et du forward du modèle hybride'''
#         q_output_size = 4
#         nb_photons = 2
#         nb_modes = 2
#         hidden_sizes = [16, 16]
#         final_output_size = 8  # Ex: 4 variables d'état * 2 actions (CartPole)

#         model = HybridMLPModel(q_output_size, nb_photons, nb_modes, hidden_sizes, final_output_size)

#         # Passage avant pour vérifier que les graphes ne crashent pas
#         out = model()

#         self.assertEqual(out.shape[-1], final_output_size)
#         self.assertFalse(torch.isnan(out).any(), "La sortie contient des valeurs NaN")

#     def test_rl_agent_forward(self):
#         '''Test de la politique linéaire (génération des actions)'''
#         state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
#         weights = torch.ones(8)  # 4 inputs * 2 outputs

#         logits = rl_agent_forward(state, weights, input_dim=4, output_dim=2)

#         # Les dimensions de sortie doivent correspondre (batch_size=1, actions=2)
#         self.assertEqual(logits.shape, (1, 2))

#     def test_compute_discounted_returns(self):
#         '''Test de la fonction utilitaire des retours escomptés'''
#         rewards = [1.0, 1.0, 1.0]
#         returns = compute_discounted_returns(rewards, gamma=0.99)
#         self.assertEqual(returns.shape, (3,))

# if __name__ == "__main__":
#     unittest.main()


import unittest

import gymnasium as gym

# import minigrid
import numpy as np
import torch
import torch.nn as nn
from lib.util import (
    HybridMLPModel,
    HybridMPSModel,
    MinigridImageOnlyWrapper,
    TorchQuantumModel,
    classic_model,
    compute_discounted_returns,
    create_hybrid_model,
    rl_agent_forward,
)


class TestRLAgentForward(unittest.TestCase):
    """Test suite for rl_agent_forward function."""

    def test_rl_agent_forward_shape_cartpole(self):
        """Test that rl_agent_forward produces correct output shape for CartPole."""
        state = torch.randn(1, 4)
        weights = torch.randn(8)
        output = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        self.assertEqual(output.shape, (1, 2))

    def test_rl_agent_forward_shape_custom(self):
        """Test rl_agent_forward with custom dimensions."""
        state = torch.randn(1, 10)
        weights = torch.randn(50)
        output = rl_agent_forward(state, weights, input_dim=10, output_dim=5)
        self.assertEqual(output.shape, (1, 5))

    def test_rl_agent_forward_no_nan(self):
        """Test that rl_agent_forward output contains no NaN values."""
        state = torch.randn(1, 4)
        weights = torch.randn(8)
        output = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        self.assertFalse(torch.isnan(output).any())


class TestComputeDiscountedReturns(unittest.TestCase):
    """Test suite for compute_discounted_returns function."""

    def test_return_shape(self):
        """Test that computed returns have correct shape."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = compute_discounted_returns(rewards, gamma=0.99)
        self.assertEqual(len(returns), len(rewards))

    def test_return_type(self):
        """Test that computed returns are torch tensors."""
        rewards = [1.0, 2.0, 3.0]
        returns = compute_discounted_returns(rewards)
        self.assertIsInstance(returns, torch.Tensor)

    def test_return_normalization(self):
        """Test that returns are normalized (mean ~0, std ~1)."""
        rewards = [1.0] * 100
        returns = compute_discounted_returns(rewards, gamma=0.99)
        mean_val = returns.mean().item()
        std_val = returns.std().item()
        self.assertAlmostEqual(mean_val, 0.0, places=1)
        self.assertGreater(std_val, 0.0)

    def test_return_no_nan(self):
        """Test that returns contain no NaN values."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        returns = compute_discounted_returns(rewards)
        self.assertFalse(torch.isnan(returns).any())


class TestMinigridWrapper(unittest.TestCase):
    """Test suite for MinigridImageOnlyWrapper."""

    def setUp(self):
        """Check if MiniGrid environment is available."""
        try:
            self.env = gym.make("MiniGrid-Empty-5x5-v0")
            self.available = True
        except gym.error.NameNotFound:
            self.available = False

    def tearDown(self):
        """Close environment if it was created."""
        if self.available and hasattr(self, "env"):
            self.env.close()

    def test_minigrid_wrapper_available(self):
        """Test MinigridImageOnlyWrapper instantiation when environment is available."""
        if not self.available:
            self.skipTest("MiniGrid environment not installed")

        wrapper = MinigridImageOnlyWrapper(self.env)
        self.assertIsNotNone(wrapper)

    def test_minigrid_wrapper_observation_space(self):
        """Test that wrapped observation space is correctly flattened."""
        if not self.available:
            self.skipTest("MiniGrid environment not installed")

        wrapper = MinigridImageOnlyWrapper(self.env)
        # MiniGrid image is typically (7, 7, 3)
        expected_flat_dim = 7 * 7 * 3
        self.assertEqual(wrapper.observation_space.shape[0], expected_flat_dim)

    def test_minigrid_wrapper_reset(self):
        """Test that wrapped environment reset produces correct flattened observation."""
        if not self.available:
            self.skipTest("MiniGrid environment not installed")

        wrapper = MinigridImageOnlyWrapper(self.env)
        obs, info = wrapper.reset()
        self.assertEqual(len(obs.shape), 1)
        self.assertEqual(obs.shape[0], 7 * 7 * 3)
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))


class TestHybridMLPModel(unittest.TestCase):
    """Test suite for HybridMLPModel."""

    def test_hybrid_mlp_initialization(self):
        """Test that HybridMLPModel initializes without errors."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        self.assertIsInstance(model, nn.Module)

    def test_hybrid_mlp_forward_pass(self):
        """Test HybridMLPModel forward pass output shape."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 8)

    def test_hybrid_mlp_no_nan(self):
        """Test that HybridMLPModel forward pass produces no NaN values."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertFalse(torch.isnan(output).any())


class TestHybridMPSModel(unittest.TestCase):
    """Test suite for HybridMPSModel."""

    def test_hybrid_mps_initialization(self):
        """Test that HybridMPSModel initializes without errors."""
        model = HybridMPSModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=3,
            bond_dim=2,
            final_output_size=8,
        )
        self.assertIsInstance(model, nn.Module)

    def test_hybrid_mps_forward_pass(self):
        """Test HybridMPSModel forward pass output shape."""
        model = HybridMPSModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=3,
            bond_dim=2,
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 8)

    def test_hybrid_mps_no_nan(self):
        """Test that HybridMPSModel forward pass produces no NaN values."""
        model = HybridMPSModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=3,
            bond_dim=2,
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertFalse(torch.isnan(output).any())


class TestTorchQuantumModel(unittest.TestCase):
    """Test suite for TorchQuantumModel."""

    def test_torchquantum_initialization(self):
        """Test that TorchQuantumModel initializes without errors."""
        model = TorchQuantumModel(
            q_output_size=4,
            n_qubit=4,
            q_depth=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        self.assertIsInstance(model, nn.Module)

    def test_torchquantum_forward_pass(self):
        """Test TorchQuantumModel forward pass output shape."""
        model = TorchQuantumModel(
            q_output_size=4,
            n_qubit=4,
            q_depth=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 8)

    def test_torchquantum_no_nan(self):
        """Test that TorchQuantumModel forward pass produces no NaN values."""
        model = TorchQuantumModel(
            q_output_size=4,
            n_qubit=4,
            q_depth=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertFalse(torch.isnan(output).any())


class TestClassicModel(unittest.TestCase):
    """Test suite for classic_model."""

    def test_classic_model_initialization(self):
        """Test that classic_model initializes without errors."""
        model = classic_model(
            layer_dim=[8, 16, 8],
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        self.assertIsInstance(model, nn.Module)

    def test_classic_model_forward_pass(self):
        """Test classic_model forward pass output shape."""
        model = classic_model(
            layer_dim=[8, 16, 8],
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 8)

    def test_classic_model_no_nan(self):
        """Test that classic_model forward pass produces no NaN values."""
        model = classic_model(
            layer_dim=[8, 16, 8],
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        with torch.no_grad():
            output = model()
        self.assertFalse(torch.isnan(output).any())


class TestCreateHybridModel(unittest.TestCase):
    """Test suite for create_hybrid_model factory function."""

    def test_create_hybrid_model_merlin_mlp(self):
        """Test create_hybrid_model with merlin_mlp backend."""
        cfg = {
            "backend": "merlin_mlp",
            "q_output_size": 4,
            "nb_photons": 2,
            "nb_modes": 2,
            "hidden_sizes": [16, 16],
        }
        model = create_hybrid_model(cfg, total_weights_needed=8)
        self.assertIsInstance(model, HybridMLPModel)

        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[1], 8)
        self.assertFalse(torch.isnan(output).any())

    def test_create_hybrid_model_merlin_mps(self):
        """Test create_hybrid_model with merlin_mps backend."""
        cfg = {
            "backend": "merlin_mps",
            "q_output_size": 4,
            "nb_photons": 2,
            "nb_modes": 3,
            "bond_dim": 2,
            "hidden_sizes": [16, 16],
        }
        model = create_hybrid_model(cfg, total_weights_needed=8)
        self.assertIsInstance(model, HybridMPSModel)

        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[1], 8)
        self.assertFalse(torch.isnan(output).any())

    def test_create_hybrid_model_torchquantum(self):
        """Test create_hybrid_model with torchquantum backend."""
        cfg = {
            "backend": "torchquantum",
            "q_output_size": 4,
            "n_qubit": 4,
            "q_depth": 2,
            "hidden_sizes": [16, 16],
        }
        model = create_hybrid_model(cfg, total_weights_needed=8)
        self.assertIsInstance(model, TorchQuantumModel)

        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[1], 8)
        self.assertFalse(torch.isnan(output).any())

    def test_create_hybrid_model_classic(self):
        """Test create_hybrid_model with classic backend."""
        cfg = {
            "backend": "classic",
            "layer_dim": [8, 16, 8],
            "hidden_sizes": [16, 16],
        }
        model = create_hybrid_model(cfg, total_weights_needed=8)
        self.assertIsInstance(model, classic_model)

        with torch.no_grad():
            output = model()
        self.assertEqual(output.shape[1], 8)
        self.assertFalse(torch.isnan(output).any())

    def test_create_hybrid_model_invalid_backend(self):
        """Test that create_hybrid_model raises error for unknown backend."""
        cfg = {"backend": "unknown_backend"}
        with self.assertRaises(ValueError):
            create_hybrid_model(cfg, total_weights_needed=8)


class TestGradientFlow(unittest.TestCase):
    """Test suite for gradient flow through hybrid models."""

    def test_hybrid_mlp_gradient_flow(self):
        """Test that gradients flow through HybridMLPModel."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        output = model()
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        self.assertTrue(has_gradients, "No gradients computed for HybridMLPModel")

    def test_hybrid_mps_gradient_flow(self):
        """Test that gradients flow through HybridMPSModel."""
        model = HybridMPSModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=3,
            bond_dim=2,
            final_output_size=8,
        )
        output = model()
        loss = output.sum()
        loss.backward()

        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        self.assertTrue(has_gradients, "No gradients computed for HybridMPSModel")

    def test_rl_agent_forward_gradient_flow(self):
        """Test that gradients flow through rl_agent_forward."""
        state = torch.randn(1, 4, requires_grad=True)
        weights = torch.randn(8, requires_grad=True)
        output = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(state.grad, "No gradient for state")
        self.assertIsNotNone(weights.grad, "No gradient for weights")


class TestBatchProcessing(unittest.TestCase):
    """Test suite for batch processing capabilities."""

    def test_rl_agent_forward_batch_size_4(self):
        """Test rl_agent_forward with batch size 4."""
        state = torch.randn(4, 4)
        weights = torch.randn(8)
        output = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        self.assertEqual(output.shape, (4, 2))

    def test_rl_agent_forward_batch_size_32(self):
        """Test rl_agent_forward with batch size 32."""
        state = torch.randn(32, 10)
        weights = torch.randn(50)
        output = rl_agent_forward(state, weights, input_dim=10, output_dim=5)
        self.assertEqual(output.shape, (32, 5))

    def test_rl_agent_forward_single_sample(self):
        """Test rl_agent_forward with single sample."""
        state = torch.randn(1, 4)
        weights = torch.randn(8)
        output = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        self.assertEqual(output.shape, (1, 2))


class TestConfigValidation(unittest.TestCase):
    """Test suite for configuration validation in create_hybrid_model."""

    def test_merlin_mlp_missing_nb_modes(self):
        """Test that merlin_mlp backend requires nb_modes."""
        cfg = {
            "backend": "merlin_mlp",
            "q_output_size": 4,
            "nb_photons": 2,
            # Missing nb_modes
            "hidden_sizes": [16, 16],
        }
        with self.assertRaises((KeyError, TypeError)):
            create_hybrid_model(cfg, total_weights_needed=8)

    def test_merlin_mps_missing_bond_dim(self):
        """Test that merlin_mps backend requires bond_dim."""
        cfg = {
            "backend": "merlin_mps",
            "q_output_size": 4,
            "nb_photons": 2,
            "nb_modes": 3,
            # Missing bond_dim
            "hidden_sizes": [16, 16],
        }
        with self.assertRaises((KeyError, TypeError)):
            create_hybrid_model(cfg, total_weights_needed=8)

    def test_classic_backend_missing_layer_dim(self):
        """Test that classic backend requires layer_dim."""
        cfg = {
            "backend": "classic",
            # Missing layer_dim
            "hidden_sizes": [16, 16],
        }
        with self.assertRaises((KeyError, TypeError)):
            create_hybrid_model(cfg, total_weights_needed=8)


class TestTrainingStep(unittest.TestCase):
    """Test suite for end-to-end training steps."""

    def test_training_step_hybrid_mlp(self):
        """Test a single training step with HybridMLPModel."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simulate one training step
        initial_params = [p.clone() for p in model.parameters()]

        weights = model()
        state = torch.randn(1, 4)
        logits = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
        loss = -logits.log_softmax(-1)[0, 0]
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        params_changed = False
        for p_old, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_old, p_new):
                params_changed = True
                break
        self.assertTrue(
            params_changed, "Model parameters did not update during training step"
        )

    def test_training_step_returns_computation(self):
        """Test computing returns and using them in a training step."""
        rewards = [1.0, 0.5, 1.5, 0.2]
        returns = compute_discounted_returns(rewards, gamma=0.99)

        # Verify returns can be used as weights
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        loss = (actions * returns[:2].view(-1, 1)).sum()
        loss.backward()

        self.assertFalse(torch.isnan(loss), "Loss contains NaN")

    def test_full_training_episode_simulation(self):
        """Simulate a full training loop for one episode."""
        model = HybridMLPModel(
            q_output_size=4,
            nb_photons=2,
            nb_modes=2,
            hidden_sizes=[16, 16],
            final_output_size=8,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simulate 5 steps of an episode
        rewards = []
        log_probs = []

        for _step in range(5):
            weights = model()
            state = torch.randn(1, 4)
            logits = rl_agent_forward(state, weights, input_dim=4, output_dim=2)
            probs = torch.softmax(logits, dim=-1)
            log_prob = torch.log(probs[0, 0])
            log_probs.append(log_prob)

            # Simulate reward
            reward = float(torch.randn(1).item())
            rewards.append(reward)

        # Compute returns and loss
        returns = compute_discounted_returns(rewards, gamma=0.99)
        loss = -(torch.stack(log_probs) * returns).sum()

        # Take optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss), "Training episode resulted in NaN loss")


if __name__ == "__main__":
    unittest.main()
