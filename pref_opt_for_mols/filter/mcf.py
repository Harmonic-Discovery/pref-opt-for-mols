import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Descriptors, MolFromSmiles, AddHs, MolToSmiles, MolFromSmarts
from rdkit.Chem.EnumerateStereoisomers import (
    GetStereoisomerCount,
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from math import log
from joblib import Parallel, delayed
import itertools

import os

RDLogger.DisableLog("rdApp.*")

_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, "mcf.csv"))
_filters = [MolFromSmarts(sm) for sm in _mcf["smarts"].tolist()]
_counts = _mcf["counts"].tolist()


def get_chiral_counts(smi, debug=False):
    """
    smi: Input smiles string
    Returns a tuple of length three:
    chiral_centers: The number of carbon chiral centers
    number of stereoisomers: the number of all possible stereoisomers
    isomeric_smiles: The isomeric smiles of the each chiral combinations
    """
    if not debug:
        RDLogger.DisableLog("rdApp.*")
    # Get the number of stereoisomers
    m = MolFromSmiles(smi)
    number_of_stereoisomers = GetStereoisomerCount(m)

    # Get the isomeric smiles that exists
    opts = StereoEnumerationOptions(tryEmbedding=True)
    m = MolFromSmiles(smi)
    isomers = tuple(EnumerateStereoisomers(m, options=opts))
    iso_smiles = [
        smi for smi in sorted(MolToSmiles(x, isomericSmiles=True) for x in isomers)
    ]

    # Number of chiral centers:
    chiral_centers = log(number_of_stereoisomers, 2)

    # Chiral carbons:
    # BUG: iso_smiles can be empty, which errors the max operator
    # when this happens, it will be caught in the filter_smiles method
    chiral_c = max([x.count("C@") for x in iso_smiles])

    # Check if chiral centers are only carbons, if yes (output: chiral_centers); else (output: chiral_c)
    if chiral_c != chiral_centers:
        ccenters = chiral_c
    else:
        ccenters = chiral_centers

    return ccenters, number_of_stereoisomers, iso_smiles


def filter_mcf(
    smiles_list,
    scaffold=None,
    n_jobs=1,
    debug=False,
):
    """
    Perform filtering of a list of smiles, optionally in parallel

    Filters currently implemented:
        - check if smiles is valid
        - check if smiles contain substruct match with given scaffold
        - check if rings < 8
        - check if molecular weight in [300, 600]
        - check if # chiral centers < 2
        - check if smiles contains any of the manually-curated smarts filters (contained in _mcf.csv)

    Args:
        scaffold (optional): smarts string representing scaffold
        n_jobs (int, default=1): number of threads to use to perform filtering in parallel
    """
    if not debug:
        RDLogger.DisableLog("rdApp.*")

    def filt(smiles, scaff):
        passes = []
        if scaff is not None:
            scaff = MolFromSmarts(scaff)
        for smi in smiles:
            try:
                mol = MolFromSmiles(smi)

                # check if valid
                if mol is None:
                    passes.append(False)
                    continue

                # check if it has the desired scaffold
                if (scaff is not None) and (not mol.HasSubstructMatch(scaff)):
                    passes.append(False)
                    continue

                # check number of rings
                ring_info = mol.GetRingInfo()
                if ring_info.NumRings() != 0 and any(
                    len(x) >= 8 for x in ring_info.AtomRings()
                ):
                    passes.append(False)
                    continue

                # check molecular weight
                mol_wt = Descriptors.ExactMolWt(mol)
                if mol_wt < 300 or mol_wt > 600:
                    passes.append(False)
                    continue

                # enforce number of chiral centers < 2
                n_chiral, _, _ = get_chiral_counts(smi)
                if n_chiral >= 2:
                    passes.append(False)
                    continue

                # check smarts filters
                # this mimics the behavior of the SupplyNode + SmartsFilter approach but a bit cleaner
                h_mol = AddHs(mol)
                if any(
                    len(h_mol.GetSubstructMatches(smarts)) >= count
                    for smarts, count in zip(_filters, _counts)
                ):
                    passes.append(False)
                    continue
            except:
                passes.append(False)
                continue
            passes.append(True)
        return passes

    if n_jobs == 1:
        return filt(smiles_list, scaffold)
    else:
        n_jobs = int(n_jobs)
        assert n_jobs > 1
        out = Parallel(n_jobs=n_jobs, max_nbytes=None, prefer="threads")(
            delayed(filt)(b, scaffold) for b in np.array_split(smiles_list, n_jobs)
        )
        return list(itertools.chain.from_iterable(out))
