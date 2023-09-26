from rdkit import Chem
from rdkit.Chem import AllChem
from .utils import disable_rdkit_log

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from joblib import Parallel, delayed
import itertools


def morgan_fingerprints_from_smiles(smiles, n_workers=1, **kwargs):
    """
    Generates morgan fingerprints from a list of smiles strings

    Arguments:
        smiles (list): list of smiles strings
        as_csr (bool): whether to store the fingerprints in csr format
        radius (int): radius of morgan fingerprints
        n_bits (int): number of bits for morgan fingerprints
        n_workers (int): number of workers for extracting fingerprints in parallel

    Returns:
        fingerprints (array): array of numpy fingerprints
        indices (list): list of indices of valid original smiles. some smiles may
        cause errors when obtaining fingerprints, so we store which ones are successfully
        computed
    """
    disable_rdkit_log()
    smiles = np.array(smiles)

    def get_fp(smi, ind):
        ixs = []
        fingerprints = []
        for ix, s in enumerate(smi):
            try:
                m = Chem.MolFromSmiles(s)
                fp = list(
                    AllChem.GetMorganFingerprintAsBitVect(
                        m,
                        radius=kwargs.get("radius", 2),
                        nBits=kwargs.get("n_bits", 1024),
                        useChirality=kwargs.get("use_chirality", False),
                    )
                )
                fingerprints.append(fp)
                ixs.append(ix)
            except:
                continue
        fingerprints = np.array(fingerprints)

        if kwargs.get("as_csr", True):
            fingerprints = csr_matrix(fingerprints)

        return fingerprints, ind[ixs]

    indices = np.arange(len(smiles))
    tmp = Parallel(n_jobs=n_workers, max_nbytes=None)(
        delayed(get_fp)(smiles[jx], jx) for jx in np.array_split(indices, n_workers)
    )
    fingerprints = sp.vstack([l for l, _ in tmp], format="csr")
    # ix contain the indices of the ligands that we were able to extract fps for
    ix = np.concatenate([i for _, i in tmp], axis=None)

    return fingerprints, ix


def _canon_smiles(smiles_list):
    assert isinstance(smiles_list, list)

    canon_smiles_list = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                canon_smiles_list.append(None)
            else:
                canon_smiles_list.append(Chem.MolToSmiles(mol))
        except:
            canon_smiles_list.append(None)
    return canon_smiles_list


def canonicalize_smiles(smiles, n_workers=1):
    disable_rdkit_log()
    assert isinstance(n_workers, int) and n_workers >= 1

    if n_workers == 1:
        return _canon_smiles(smiles)
    else:
        out = Parallel(n_jobs=n_workers, max_nbytes=None, prefer="threads")(
            delayed(_canon_smiles)(smiles_batch.tolist())
            for smiles_batch in np.array_split(smiles, n_workers)
        )
        return list(itertools.chain.from_iterable(out))


def tanimoto(X, Y=None):
    """
    computes the tanimoto kernel
    """
    if Y is None:
        Y = X
    prod = np.dot(X, Y.T)
    norm_X = np.sum(X, axis=1).reshape(-1, 1)
    norm_Y = np.sum(Y, axis=1).reshape(-1, 1)
    return (prod / (norm_X + norm_Y.T - prod)).toarray()


def internal_diversity(smiles, n_workers=1):
    """
    Computes the internal diversity, defined by

    1 - AverageTanimoto(Mols)

    Here we use 1024-bit ECFP4 fingerprints
    """
    fps, _ = morgan_fingerprints_from_smiles(smiles, n_workers=n_workers, as_csr=True)
    tc = tanimoto(fps)
    d = tc.shape[0]
    tc = ((tc - np.eye(d)).sum()) / (d * (d - 1))
    return 1.0 - tc


def frac_valid(smiles, n_workers=1):
    """
    Computes the fraction of smiles that represent valid molecules
    """
    smiles = canonicalize_smiles(smiles, n_workers=n_workers)
    return 1.0 - sum([smi is None for smi in smiles]) / len(smiles)


def frac_unique(smiles, n_workers=1):
    """
    Computes the fraction of unique smiles strings in a list of smiles
    """
    smiles = canonicalize_smiles(smiles, n_workers=n_workers)
    return len(list(set(smiles))) / len(smiles)


def strip_invalid(smiles, n_workers=1):
    smiles = canonicalize_smiles(smiles, n_workers=n_workers)
    return [smi for smi in smiles if smi is not None]


def fcd_score(gen_smiles, ref_smiles, device="cpu", n_workers=8):
    """
    Computes the Frechet ChemNet Distance

    REQUIRES `fcd_torch` TO BE INSTALLED
    """
    try:
        from fcd_torch import FCD
    except:
        raise Exception(
            "Could not import package fcd_torch, try `pip install fcd_torch`"
        )

    valid_gen_smiles = strip_invalid(gen_smiles, n_workers=n_workers)
    valid_ref_smiles = strip_invalid(ref_smiles, n_workers=n_workers)

    fcd = FCD(device=device, n_jobs=n_workers)
    score = fcd(valid_gen_smiles, valid_ref_smiles)
    return score


def frac_contains_scaffold(smiles, scaffold_smiles):
    disable_rdkit_log()
    scaffold_mol = Chem.MolFromSmarts(scaffold_smiles)
    smiles = strip_invalid(smiles)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    return sum([m.HasSubstructMatch(scaffold_mol) for m in mols]) / len(smiles)
