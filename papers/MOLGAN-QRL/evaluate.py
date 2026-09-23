# import torch
# import torch.nn.functional as F
# from rdkit import Chem
# from rdkit.Chem import Draw
# from chemical_reward import evaluate_and_reward, ATOM_TYPES, BOND_TYPES
# from WGAN import Generator  # Import manquant ajouté

# # 1. Configuration 
# z_dim = 8
# N_nodes = 9
# N_atoms = 5
# N_bonds = 5
# num_samples = 497

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 2. Initialisation et chargement du Générateur (Lignes décommentées)
# G = Generator([128, 256, 512], z_dim, N_nodes, N_bonds, N_atoms, 0.0).to(device)

# # VÉRITÉ TECHNIQUE : Ce script nécessite le fichier des poids sauvegardés.
# try:
#     G.load_state_dict(torch.load("generator_final.pth", map_location=device))
#     print("Poids du modèle chargés avec succès.")
# except FileNotFoundError:
#     print("ERREUR : Le fichier 'generator_final.pth' est introuvable.")
#     print("Si tu n'as pas ajouté torch.save(G.state_dict(), 'generator_final.pth') à la fin de main.py, les poids de tes 200 époques ont été perdus en mémoire.")
#     exit()

# print(f"Génération de {num_samples} molécules en cours...")
# G.eval()

# with torch.no_grad():
#     z = torch.randn(num_samples, z_dim).to(device)
#     edges_logits, nodes_logits = G(z)
    
#     fake_adj = F.gumbel_softmax(edges_logits, tau=1.0, hard=True, dim=-1)
#     fake_nodes = F.gumbel_softmax(nodes_logits, tau=1.0, hard=True, dim=-1)

# adj_discrete = torch.argmax(fake_adj, dim=-1).cpu().numpy()
# nodes_discrete = torch.argmax(fake_nodes, dim=-1).cpu().numpy()

# valid_mols = []
# valid_smiles = []

# for i in range(num_samples):
#     mol = Chem.RWMol()
#     node_indices = []
    
#     for atom_idx in nodes_discrete[i]:
#         atom_symbol = ATOM_TYPES[atom_idx]
#         idx = mol.AddAtom(Chem.Atom(atom_symbol))
#         node_indices.append(idx)
        
#     num_atoms = len(node_indices)
#     for j in range(num_atoms):
#         for k in range(j + 1, num_atoms):
#             bond_type_idx = adj_discrete[i, j, k]
#             if bond_type_idx > 0: 
#                 bond = BOND_TYPES.get(bond_type_idx)
#                 if bond:
#                     try:
#                         mol.AddBond(node_indices[j], node_indices[k], bond)
#                     except Exception:
#                         pass 
                        
#     try:
#         Chem.SanitizeMol(mol)
#         smiles = Chem.MolToSmiles(mol)
#         valid_smiles.append(smiles)
#         valid_mols.append(mol)
#     except Exception:
#         pass

# validity_score = (len(valid_smiles) / num_samples) * 100
# unique_smiles = set(valid_smiles)
# uniqueness_score = (len(unique_smiles) / len(valid_smiles)) * 100 if len(valid_smiles) > 0 else 0.0
# nvu_score = len(unique_smiles)

# print("\n--- RÉSULTATS FINAUX MOLGAN-QRL ---")
# print(f"Validité   : {validity_score:.2f}%")
# print(f"Unicité    : {uniqueness_score:.2f}%")
# print(f"Score NVU  : {nvu_score} molécules uniques et valides")

# if nvu_score > 0:
#     unique_mols = [Chem.MolFromSmiles(s) for s in list(unique_smiles)[:16]]
#     img = Draw.MolsToGridImage(unique_mols, molsPerRow=4, subImgSize=(200, 200), returnPNG=False)
#     img.save("generated_molecules.png")
#     print("\nUne image des structures générées a été sauvegardée sous 'generated_molecules.png'.")

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
from chemical_reward import evaluate_and_reward, ATOM_TYPES, BOND_TYPES
from WGAN import Generator

# 1. Configuration 
z_dim = 8
N_nodes = 9
N_atoms = 5
N_bonds = 5
num_samples = 497

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chargement du Générateur
G = Generator([128, 256, 512], z_dim, N_nodes, N_bonds, N_atoms, 0.0).to(device)

try:
    G.load_state_dict(torch.load("generator_final.pth", map_location=device))
    print("Poids du modèle chargés avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier 'generator_final.pth' est introuvable.")
    exit()

print(f"Génération de {num_samples} molécules en cours...")
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
        
        # Calcul du score QED (Drug-Likeliness)
        score_qed = qed(mol)
        
        # Filtrage strict : on ne garde que les uniques valides avec un QED >= 0.5
        if smiles not in valid_smiles and score_qed >= 0.5:
            valid_smiles.append(smiles)
            valid_mols.append(mol)
            qed_scores.append(score_qed)
            
    except Exception:
        pass

validity_score = (len(valid_smiles) / num_samples) * 100
nvu_score = len(valid_smiles)

print("\n--- RÉSULTATS FILTRÉS (QED >= 0.5) ---")
print(f"Nombre de molécules uniques, valides ET à fort potentiel (QED >= 0.5) : {nvu_score}")
if nvu_score > 0:
    print(f"Score QED moyen pour ces molécules : {sum(qed_scores)/len(qed_scores):.3f}")

# Sauvegarde des meilleures structures filtrées
if nvu_score > 0:
    # Trie par meilleur score QED pour afficher les plus prometteuses en premier
    sorted_pairs = sorted(zip(valid_mols, qed_scores), key=lambda x: x[1], reverse=True)
    best_mols = [item[0] for item in sorted_pairs[:16]]
    
    img = Draw.MolsToGridImage(best_mols, molsPerRow=4, subImgSize=(200, 200), returnPNG=False)
    img.save("best_drug_candidates.png")
    print("\nUne image des meilleures molécules filtrées a été sauvegardée sous 'best_drug_candidates.png'.")
else:
    print("\nAucune molécule n'a atteint le seuil de QED >= 0.5 parmi les échantillons valides (ce qui est normal pour des fragments de type QM9).")