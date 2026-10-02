"""SMILES -> PyG graph (for the GNN) and Morgan fingerprint (for baselines)."""
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch_geometric.data import Data

NUM_ATOM_FEATURES = 6


def morgan(smiles, n_bits=1024, radius=2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"bad SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES: {smiles}")
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)
# ponytail: raw atomic numbers as float features are fine for a smoke test;
# for the real model, embed atom types / one-hot instead. add when accuracy plateaus.
