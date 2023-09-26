import argparse
import pandas as pd
from pref_opt_for_mols.models import GPTLightning, CharRNNLightning
from pref_opt_for_mols.dataset import NextTokenSmilesDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
import torch
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arch",
    type=str,
    default="rnn",
    help="which model class to use, either 'gpt' or 'rnn'",
)
parser.add_argument(
    "--config",
    required=True,
    type=str,
)
parser.add_argument(
    "--device",
    default=0,
    type=int,
)

args = parser.parse_args()


def train(model, config, train_loader, test_loader=None):
    logger = pl.loggers.NeptuneLogger(
        project="harmonic-discovery/HD-Generative-Models",
        api_token=os.environ.get("NEPTUNE_API_KEY"),
    )

    # log configs
    for key, value in config.items():
        logger.experiment[f"config/{key}"].log(value)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get("model_path"),
        filename="model",
        every_n_train_steps=100,
    )
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=config.get("model_path"),
        filename="model_best",
        monitor="val_loss",
        mode="min",
    )

    # create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=config.get("max_epochs", 10),
        logger=logger,
        default_root_dir=config.get("model_path"),
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, checkpoint_callback_best],
        val_check_interval=0.2,
    )

    trainer.fit(model, train_loader, test_loader)
    return model, logger


def load_moses(split="train"):
    assert split in ["train", "test"]
    return pd.read_csv(f"data/{split}.csv")["smiles"].tolist()


if __name__ == "__main__":
    with open(args.config) as fp:
        config = json.load(fp)

    assert config.get("model_path") is not None

    os.makedirs(config.get("model_path"), exist_ok=True)

    # smiles_col = "selfies" if config.get("selfies", False) else "smiles"
    # scaffold_col = "scaffold_selfies" if config.get("selfies", False) else "scaffold"
    # train_set = pd.read_csv(args.train_smiles)
    # train_smiles = train_set[smiles_col].tolist()
    # train_scaffolds = train_set[scaffold_col].tolist()
    # del train_set

    # if args.test_smiles is not None:
    #     test_set = pd.read_csv(args.test_smiles)
    #     test_smiles = test_set[smiles_col].tolist()
    #     test_scaffolds = test_set[scaffold_col].tolist()
    #     del test_set
    # else:
    #     test_smiles, test_scaffolds = None, None

    train_smiles = load_moses("train")
    test_smiles = load_moses("test")

    train_dataset = NextTokenSmilesDataset(
        train_smiles,
        enclose=True,
        seq_len=config.get("seq_len"),
        aug_prob=config.get("aug_prob", 0.0),
    )

    test_dataset = NextTokenSmilesDataset(
        test_smiles,
        enclose=True,
        seq_len=config.get("seq_len"),
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        persistent_workers=True,
    )
    print(
        f"train dataset size: {len(train_dataset)}, train dataloader size: {len(train_loader)}"
    )

    if args.arch == "gpt":
        config["vocab_size"] = len(train_dataset.tokenizer.vocabulary)
        config["block_size"] = config.get("seq_len") - 1
        model = GPTLightning(config)
    elif args.arch == "rnn":
        model = CharRNNLightning(config)
    else:
        raise ValueError(f"Unrecognized model {args.arch}")

    with open(os.path.join(config.get("model_path"), "config.json"), "w") as fp:
        json.dump(config, fp)

    model, logger = train(model, config, train_loader, test_loader)
