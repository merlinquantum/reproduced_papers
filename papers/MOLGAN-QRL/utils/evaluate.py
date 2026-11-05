import torch
import torch.nn.functional as F
from lib.chemical_reward import ATOM_TYPES, BOND_TYPES
from lib.WGAN import Generator
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed

# 1. Configuration
z_dim = 8
N_nodes = 9
N_atoms = 5
N_bonds = 5
num_samples = 497

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator([128, 256, 512], z_dim, N_nodes, N_bonds, N_atoms, 0.0).to(device)

try:
    G.load_state_dict(torch.load("generator_final.pth", map_location=device))
    print("Model's weight loaded with success.")
except FileNotFoundError:
    print("ERROR : file 'generator_final.pth' not in the directory")
    exit()

print(f"Generation {num_samples} molecules...")
G.eval()

with torch.no_grad():
    z = torch.randn(num_samples, z_dim).to(device)
    edges_logits, nodes_logits = G(z)

    fake_adj = F.gumbel_softmax(edges_logits, tau=1.0, hard=True, dim=-1)
    fake_nodes = F.gumbel_softmax(nodes_logits, tau=1.0, hard=True, dim=-1)

adj_discrete = torch.argmax(fake_adj, dim=-1).cpu().numpy()
nodes_discrete = torch.argmax(fake_nodes, dim=-1).cpu().numpy()

valid_mols = []
valid_smiles = []
qed_scores = []

for i in range(num_samples):
    mol = Chem.RWMol()
    node_indices = []

    for atom_idx in nodes_discrete[i]:
        atom_symbol = ATOM_TYPES[atom_idx]
        idx = mol.AddAtom(Chem.Atom(atom_symbol))
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
        smiles = Chem.MolToSmiles(mol)

        score_qed = qed(mol)

        if smiles not in valid_smiles and score_qed >= 0.5:
            valid_smiles.append(smiles)
            valid_mols.append(mol)
            qed_scores.append(score_qed)

    except Exception:
        pass

validity_score = (len(valid_smiles) / num_samples) * 100
nvu_score = len(valid_smiles)

print("\n--- Results (QED >= 0.5) ---")
print(f"Number of unique, valid and drug likely molecules (QED >= 0.5) : {nvu_score}")
if nvu_score > 0:
    print(
        f"mean of the scores of those molecules : {sum(qed_scores) / len(qed_scores):.3f}"
    )

if nvu_score > 0:
    sorted_pairs = sorted(zip(valid_mols, qed_scores), key=lambda x: x[1], reverse=True)
    best_mols = [item[0] for item in sorted_pairs[:16]]

    img = Draw.MolsToGridImage(
        best_mols, molsPerRow=4, subImgSize=(200, 200), returnPNG=False
    )
    img.save("best_drug_candidates.png")
    print(
        "\n An image of the best filtered molecule have been loaded on : 'best_drug_candidates.png'."
    )
else:
    print("\n No molecules have achieve a drug-likelyness score upper than 0.5.")
