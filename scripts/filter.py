import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial
from pref_opt_for_mols.filter import filter_mcf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smiles",
    required=True,
    type=str,
    help="path to smiles csv generated with sample.py"
) 
parser.add_argument(
    "--out",
    required=True,
    type=str,
    help="path to output csv"
)
parser.add_argument(
    "--method",
    default="mcf",
    type=str,
    help="filtering method to use (supported: 'mcf')"
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="batch size to use for filtering"
)
args = parser.parse_args()


def batch_iter(smiles, batch_size):
    for i in range(0, len(smiles), batch_size):
        yield smiles[i:i+batch_size]


if __name__ == "__main__":
    smiles = pd.read_csv(args.smiles)["smiles"]
    supplier = batch_iter(smiles, args.batch_size)

    if args.method == "mcf":
        filterer = partial(filter_mcf, n_jobs=64)
    else:
        raise ValueError(f"Unrecognized method {args.method}")

    label = []
    print(f"[FILTER] total smiles: {len(smiles)}, "
          f"filtering method: {args.method}, "
          f"batch size: {args.batch_size}")
    for batch in tqdm(supplier, total=int(round(len(smiles) / args.batch_size))):
        batch_passes = filterer(batch)
        label.extend(batch_passes)

    df = pd.DataFrame({"smiles": smiles, "label": label})
    df.to_csv(args.out, index=False)
