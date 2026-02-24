# see-ii.py
# Interferometer-Interferometer PINN for the damped oscillator using oscillator_core + merlin_quantum

from datetime import datetime
import csv

import torch
import torch.nn as nn

from ..config import SEE_N_EPOCHS, SEE_LR, SEE_PLOT_EVERY, DTYPE
from ..utils import make_time_grid, make_optimizer
from .core_see import train_see
from ..layer_merlin import make_interf_qlayer, BranchMerlin


# ============================================================
#  II_PINN model: two MerLin quantum branches
# ============================================================


class II_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, n_photons: int) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons), n_outputs=3
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons), n_outputs=3
        )

        # Fusion head: combines outputs of both branches into (rho, u, p)
        self.fusion = nn.Sequential(
            nn.Linear(3, 8, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=DTYPE),
        )

        # Human-readable size label (e.g. "2", "3", "4")
        self.size_label = f"{n_photons}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum two quantum branches then apply fusion head
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        combined = out1 + out2  # [N, 3]
        return self.fusion(combined)  # [N, 3]

    # def forward(self, t: torch.Tensor) -> torch.Tensor:
    #     # Forward pass: sum of the two interferometer branches
    #     return self.branch1(t) + self.branch2(t)


MODELS = [
    ("2", 2),
    ("3", 3),
    ("4", 4),
]


# def run() -> None:
#     """Run the Interferometer–Interferometer DHO PINN experiment."""
#     torch.manual_seed(0)
#     np.random.seed(0)

#     model = II_PINN()
#     train_see(
#         model=model,
#         t_train=make_time_grid(),
#         optimizer=make_optimizer(model, lr=SEE_LR),
#         n_epochs=SEE_N_EPOCHS,
#         plot_every=SEE_PLOT_EVERY,
#         out_dir="HQPINN/SEE/results",
#         model_label="Interferometer-Interferometer",
#     )


def run():
    """Run all SEE Interferometer-Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = f"HQPINN/SEE/results/ii_summary_{timestamp}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Size",
                "Trainable parameters",
                "Loss",
                "Density error",
                "Pressure error",
            ]
        )

        for label, n_photons in MODELS:
            print(f"\nTraining SEE-II {n_photons} photons")

            model = II_PINN(n_photons=n_photons)
            optimizer = make_optimizer(model, lr=SEE_LR)

            final_loss, err_rho, err_p, n_params = train_see(
                model=model,
                t_train=make_time_grid(),  # kept for API consistency
                optimizer=optimizer,
                n_epochs=SEE_N_EPOCHS,
                plot_every=SEE_PLOT_EVERY,
                out_dir=f"HQPINN/SEE/results/ii-{label}",
                model_label=f"ii-{label}",
            )

            writer.writerow(
                [
                    "ii",  # model type: Interferometer-Interferometer
                    label,  # size label ("2", "3", "4")
                    n_params,
                    f"{final_loss:.6e}",
                    f"{err_rho:.6e}",
                    f"{err_p:.6e}",
                ]
            )

    print(f"Summary CSV saved to: {out_csv}")
