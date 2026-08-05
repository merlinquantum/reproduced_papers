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
import minigrid
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
        self.env = gym.make("MiniGrid-Empty-5x5-v0")
        self.available = True

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


if __name__ == "__main__":
    unittest.main()
