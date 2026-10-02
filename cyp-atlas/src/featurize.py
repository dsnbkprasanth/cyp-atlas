"""SMILES -> Morgan fingerprint (numpy int8)."""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

N_BITS = 1024


def morgan(smiles, n_bits=N_BITS, radius=2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"bad SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_safe(smiles):
    """None on failure instead of raising — for bulk screens over messy files."""
    try:
        return morgan(smiles)
    except Exception:
        return None
