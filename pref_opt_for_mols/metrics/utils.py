from rdkit import rdBase
from rdkit.Chem import Descriptors, MolFromSmiles, AddHs, MolToSmiles, MolFromSmarts
from rdkit.Chem.EnumerateStereoisomers import (
    GetStereoisomerCount,
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
import math


def disable_rdkit_log():
    rdBase.DisableLog("rdApp.*")


def get_chiral_counts(smi, debug=False):
    """
    smi: Input smiles string
    Returns a tuple of length three:
    chiral_centers: The number of carbon chiral centers
    number of stereoisomers: the number of all possible stereoisomers
    isomeric_smiles: The isomeric smiles of the each chiral combinations
    """
    disable_rdkit_log()
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
    chiral_centers = math.log(number_of_stereoisomers, 2)

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
