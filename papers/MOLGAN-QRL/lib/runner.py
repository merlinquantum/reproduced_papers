import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lib.chemical_reward import evaluate_and_reward
from lib.quantum_layer import PhotonicRewardModule
from lib.WGAN import Discriminator, Generator, gradient_penalty
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj, to_dense_batch


def get_default_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "defaults.json"
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def prepare_data(cfg: dict):
    print(f"--- loading of QM9 ({cfg['data_dir']}) ---")
    dataset = QM9(root=cfg["data_dir"])
    n_nodes = cfg["n_nodes"]

    node_tensors, adj_tensors = [], []
    print(f"extraction of {cfg['target_sample_count']} valid molecules...")

    for data in dataset:
        if data.num_nodes > n_nodes:
            continue

        x_heavy = data.x[:, 1:5]
        x_dense, mask = to_dense_batch(x_heavy, max_num_nodes=n_nodes)
        x_dense = x_dense[0]

        is_empty = (~mask[0]).float().unsqueeze(1)
        x_final = torch.cat([is_empty, x_dense], dim=1)

        adj_dense = to_dense_adj(
            data.edge_index, edge_attr=data.edge_attr, max_num_nodes=n_nodes
        )[0]
        has_bond = adj_dense.sum(dim=-1)
        no_bond = (has_bond == 0).float().unsqueeze(-1)
        no_bond.diagonal().fill_(1.0)

        adj_final = torch.cat([no_bond, adj_dense], dim=-1)

        node_tensors.append(x_final)
        adj_tensors.append(adj_final)

        if len(node_tensors) >= cfg["target_sample_count"]:
            break

    stacked_nodes = torch.stack(node_tensors)
    stacked_adjs = torch.stack(adj_tensors)

    dataloader = DataLoader(
        TensorDataset(stacked_nodes, stacked_adjs),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    return dataloader, stacked_nodes, stacked_adjs


def train_and_evaluate(*args, **kwargs):
    """
    Training entry point
    wait for (project_dir, config), (config, ...), or can be called without args.
    """
    cfg = get_default_config()

    for arg in args:
        if isinstance(arg, dict):
            cfg.update(arg)
        elif hasattr(arg, "__dict__"):
            cfg.update(vars(arg))
        elif isinstance(arg, (str, Path)):
            path_arg = Path(arg)
            if (path_arg / "configs" / "defaults.json").exists():
                cfg["project_dir"] = str(path_arg)
            else:
                cfg["output_dir"] = str(path_arg)

    if kwargs:
        cfg.update({k: v for k, v in kwargs.items() if v is not None})

    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on : {device}")

    dataloader, stacked_nodes, stacked_adjs = prepare_data(cfg)

    os.makedirs(cfg["log_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["log_dir"])

    n_nodes = cfg["n_nodes"]
    n_atoms = cfg["n_atoms"]
    n_bonds = cfg["n_bonds"]
    z_dim = cfg["z_dim"]
    d_conv_dims = (cfg["d_conv_dims"][0], cfg["d_conv_dims"][1], cfg["d_conv_dims"][2])

    G = Generator(
        cfg["g_conv_dims"], z_dim, n_nodes, n_bonds, n_atoms, cfg["g_dropout"]
    ).to(device)

    D = Discriminator(
        d_conv_dims, m_dim=n_atoms, b_dim=n_bonds - 1, dropout_rate=cfg["d_dropout"]
    ).to(device)

    print("ACP Computation...")
    sample_adj = stacked_adjs.flatten(start_dim=1).cpu().numpy()
    sample_nodes = stacked_nodes.flatten(start_dim=1).cpu().numpy()
    sample_x = np.concatenate([sample_adj, sample_nodes], axis=1)

    pca = PCA(n_components=cfg["pca_components"])
    pca.fit(sample_x)

    Q = PhotonicRewardModule(
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        num_atoms=n_nodes,
        num_bonds=n_bonds,
        num_atom_types=n_atoms,
        nb_modes=cfg["nb_modes"],
    ).to(device)

    opt_g = torch.optim.RMSprop(G.parameters(), lr=cfg["lr_g"])
    opt_d = torch.optim.RMSprop(D.parameters(), lr=cfg["lr_d"])
    opt_q = torch.optim.RMSprop(Q.parameters(), lr=cfg["lr_q"])

    epochs = cfg["epochs"]
    n_critic = cfg["n_critic"]
    lambda_gp = cfg["lambda_gp"]
    tau = cfg["tau"]
    warmup_epochs = cfg["warmup_epochs"]
    ramp_epochs = cfg["ramp_epochs"]
    lambda_max = cfg["lambda_max"]
    ema_beta = cfg["ema_beta"]

    reward_ema = 0.0
    global_step = 0
    val_ratio = 0.0
    uniq_ratio = 0.0

    print("\n--- Training of Mol-GAN QRL ---")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        if epoch <= warmup_epochs:
            current_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / ramp_epochs
            current_lambda = min(lambda_max, progress * lambda_max)

        for batch_nodes, batch_adj in dataloader:
            batch_size = batch_nodes.size(0)
            batch_nodes, batch_adj = batch_nodes.to(device), batch_adj.to(device)

            for _ in range(n_critic):
                opt_d.zero_grad()
                z = torch.randn(batch_size, z_dim, device=device)

                with torch.no_grad():
                    edges_logits, nodes_logits = G(z)
                    fake_adj = F.gumbel_softmax(
                        edges_logits, tau=tau, hard=True, dim=-1
                    )
                    fake_nodes = F.gumbel_softmax(
                        nodes_logits, tau=tau, hard=True, dim=-1
                    )

                real_val = D(batch_adj, None, batch_nodes)
                fake_val = D(fake_adj, None, fake_nodes)

                gp = gradient_penalty(
                    D, batch_nodes, batch_adj, fake_nodes, fake_adj, device
                )
                d_loss = -torch.mean(real_val) + torch.mean(fake_val) + (lambda_gp * gp)

                d_loss.backward()
                opt_d.step()

            opt_q.zero_grad()
            with torch.no_grad():
                z = torch.randn(batch_size, z_dim, device=device)
                edges_logits, nodes_logits = G(z)
                fake_adj_hard = F.gumbel_softmax(
                    edges_logits, tau=tau, hard=True, dim=-1
                )
                fake_nodes_hard = F.gumbel_softmax(
                    nodes_logits, tau=tau, hard=True, dim=-1
                )
                target_rc_fake, val_ratio, uniq_ratio = evaluate_and_reward(
                    fake_adj_hard, fake_nodes_hard
                )
                target_rc_real, _, _ = evaluate_and_reward(batch_adj, batch_nodes)

            pred_rq_fake = Q(fake_adj_hard, fake_nodes_hard)
            pred_rq_real = Q(batch_adj, batch_nodes)

            q_loss = torch.mean(
                torch.abs(pred_rq_real - target_rc_real)
                + torch.abs(pred_rq_fake - target_rc_fake)
            )
            q_loss.backward()
            opt_q.step()

            opt_g.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            edges_logits, nodes_logits = G(z)

            fake_adj_soft = F.gumbel_softmax(edges_logits, tau=tau, hard=False, dim=-1)
            fake_nodes_soft = F.gumbel_softmax(
                nodes_logits, tau=tau, hard=False, dim=-1
            )

            fake_val = D(fake_adj_soft, None, fake_nodes_soft)
            g_loss_wgan = -torch.mean(fake_val)

            pred_rq_soft = Q(fake_adj_soft, fake_nodes_soft)
            instant_reward_tensor = torch.mean(pred_rq_soft)

            if global_step == 0:
                reward_ema = instant_reward_tensor.item()
            else:
                reward_ema = (
                    ema_beta * reward_ema
                    + (1.0 - ema_beta) * instant_reward_tensor.item()
                )

            g_loss_total = (1.0 - current_lambda) * g_loss_wgan + current_lambda * (
                -instant_reward_tensor
            )
            g_loss_total.backward()
            opt_g.step()

            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator_Total", g_loss_total.item(), global_step)
            writer.add_scalar("Loss/Generator_WGAN", g_loss_wgan.item(), global_step)
            writer.add_scalar("Photonic/L1_Loss", q_loss.item(), global_step)
            writer.add_scalar(
                "Reward/Average_RDKit_Rc", target_rc_fake.mean().item(), global_step
            )
            writer.add_scalar("Reward/EMA_Quantum_Rq", reward_ema, global_step)
            writer.add_scalar("Metrics/Validity", val_ratio, global_step)
            writer.add_scalar("Metrics/Uniqueness", uniq_ratio, global_step)
            writer.add_scalar("Hyperparameters/Lambda_RL", current_lambda, global_step)

            global_step += 1

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"D_Loss: {d_loss.item():.4f} | "
            f"G_Loss: {g_loss_total.item():.4f} | "
            f"Val: {val_ratio:.2f} | "
            f"Uniq: {uniq_ratio:.2f} | "
            f"Lambda: {current_lambda:.2f} | "
            f"Time: {elapsed:.2f}s"
        )
        torch.save(G.state_dict(), cfg["save_model_path"])

    writer.close()
    print(f"Train complete. Model saved as : {cfg['save_model_path']}")

    return {
        "status": "success",
        "saved_model": cfg.get("save_model_path"),
        "final_validity": val_ratio,
        "final_uniqueness": uniq_ratio,
    }


run = train_and_evaluate
main = train_and_evaluate


if __name__ == "__main__":
    train_and_evaluate()
