# run_cc_oscillator.py

import torch

from oscillator_core import train_oscillator_pinn
from mlp_branches import CC_PINN


def main():
    # Hyperparameters
    lr = 0.002
    n_epochs = 2000
    plot_every = 100

    # Torch setup and time grid
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cpu")
    torch.manual_seed(0)

    t_train = torch.linspace(0.0, 1.0, 200, device=device).reshape(-1, 1)

    model = CC_PINN(dtype=dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_oscillator_pinn(
        model=model,
        t_train=t_train,
        optimizer=optimizer,
        n_epochs=n_epochs,
        plot_every=plot_every,
        out_dir="HQPINN/results",
        model_label="classical-classical",
    )


if __name__ == "__main__":
    main()
