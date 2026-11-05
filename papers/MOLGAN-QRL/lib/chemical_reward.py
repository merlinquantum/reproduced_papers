import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.*")

ATOM_TYPES = ["C", "N", "O", "F", "P"]
BOND_TYPES = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def evaluate_and_reward(adj_tensor, node_matrix):
    """
    Evaluates a batch of discrete molecular graphs using RDKit to compute chemical
    properties (QED score), validity rates, and uniqueness metrics.

    Args:
        adj_tensor (torch.Tensor): Encoded adjacency tensor of shape
            (batch_size, num_nodes, num_nodes, bond_types).
        node_matrix (torch.Tensor): Encoded node feature matrix of shape
            (batch_size, num_nodes, atom_types).

    Returns:
        tuple:
            - reward_tensor (torch.Tensor): Tensor of computed RDKit QED scores of shape (batch_size, 1).
            - valid_ratio (float): Ratio of chemically valid molecules in the batch.
            - unique_ratio (float): Ratio of unique valid molecules among all valid ones.
    """
    batch_size = adj_tensor.shape[0]
    rc_scores = []
    valid_smiles = []

    adj_discrete = torch.argmax(adj_tensor, dim=-1).cpu().numpy()
    nodes_discrete = torch.argmax(node_matrix, dim=-1).cpu().numpy()

    for i in range(batch_size):
        mol = Chem.RWMol()
        node_indices = []

        for atom_idx in nodes_discrete[i]:
            atom_symbol = ATOM_TYPES[atom_idx]
            new_atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(new_atom)
            node_indices.append(idx)

        num_atoms = len(node_indices)
        for j in range(num_atoms):
            for k in range(j + 1, num_atoms):
                bond_type_idx = adj_discrete[i, j, k]
                if bond_type_idx > 0:
                    bond = BOND_TYPES.get(bond_type_idx)
                    if bond:
                        try:
                            mol.AddBond(node_indices[j], node_indices[k], bond)
                        except Exception:
                            pass

        try:
            Chem.SanitizeMol(mol)
            score = QED.default(mol)
            smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(smiles)
        except Exception:
            score = 0.0

        rc_scores.append([score])

    valid_ratio = len(valid_smiles) / batch_size
    unique_ratio = (
        len(set(valid_smiles)) / len(valid_smiles) if len(valid_smiles) > 0 else 0.0
    )

    reward_tensor = torch.tensor(
        rc_scores, dtype=torch.float32, device=adj_tensor.device
    )
    return reward_tensor, valid_ratio, unique_ratio
