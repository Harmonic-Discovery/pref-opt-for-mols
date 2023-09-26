from pref_opt_for_mols.metrics import morgan_fingerprints_from_smiles, strip_invalid
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smiles",
    type=str,
    default="rnn",
    help="which model class to use, either 'gpt' or 'rnn'",
)
parser.add_argument(
    "--clf_path",
    required=True,
    type=str,
)
parser.add_argument(
    "--out",
    required=True,
    type=str,
)
args = parser.parse_args()


def load_pkl(filepath):
    """
    Load pickle file

    Arguments:
        filepath (str): filepath to load object from

    Returns:
        object
    """
    with open(filepath, "rb") as fp:
        data = pickle.load(fp)

    return data


if __name__ == "__main__":
    smiles = pd.read_csv(args.smiles)["smiles"].tolist()
    smiles = strip_invalid(smiles)

    model = load_pkl(args.clf_path)

    fps = morgan_fingerprints_from_smiles(smiles)[0].toarray()
    labels = model.predict(fps)
    scores = model.predict_proba(fps)
    print(scores.shape)
    print(scores[:, 1].flatten()[:10])

    pd.DataFrame(
        {"smiles": smiles, "label": labels, "predict_prob": scores[:, 1]}
    ).to_csv(args.out)
