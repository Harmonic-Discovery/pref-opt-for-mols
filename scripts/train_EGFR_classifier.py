from pref_opt_for_mols.metrics import morgan_fingerprints_from_smiles
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

RANDOM_STATE = 42


def save_pkl(data, filepath):
    """
    Save pkl file

    Arguments:
        data (obj): data to save
        filepath (str): filepath for save location
    """
    with open(filepath, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data = pd.read_csv("data/EGFR_activity_data.csv")
    train_data, test_data = train_test_split(
        data, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train = morgan_fingerprints_from_smiles(train_data["smiles"].tolist())[
        0
    ].toarray()
    y_train = train_data["is_active"].to_numpy()
    X_test = morgan_fingerprints_from_smiles(test_data["smiles"].tolist())[0].toarray()
    y_test = test_data["is_active"].to_numpy()

    print(X_test.shape)
    clf = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(
        classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["inactive", "active"]
        )
    )
    os.makedirs("checkpoints", exist_ok=True)
    save_pkl(clf, os.path.join("checkpoints", "EGFR_classifier.pkl"))
