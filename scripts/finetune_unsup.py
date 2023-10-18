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
print(f"Using device {args.device}")


def train(model, config, train_loader, test_loader=None):
    if config.get("neptune_project"):
        logger = pl.loggers.NeptuneLogger(
            project=config.get("neptune_project"),
            api_token=os.environ.get("NEPTUNE_API_KEY"),
        )
    else:
        logger = None

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


if __name__ == "__main__":
    with open(args.config) as fp:
        config = json.load(fp)

    assert config.get("model_path") is not None

    os.makedirs(config.get("model_path"), exist_ok=True)

    with open(os.path.join(config.get("pretrained_model_path"), "config.json")) as fp:
        pretrain_config = json.load(fp)

    train_smiles = pd.read_csv(config.get("train_smiles_file"))["smiles"].tolist()
    test_smiles = (
        None
        if config.get("test_smiles_file") is None
        else pd.read_csv(config.get("test_smiles_file"))["smiles"].tolist()
    )

    train_dataset = NextTokenSmilesDataset(
        train_smiles,
        enclose=True,
        seq_len=pretrain_config.get("seq_len"),
        aug_prob=config.get("aug_prob", 0.0),
    )

    test_dataset = (
        None
        if test_smiles is None
        else NextTokenSmilesDataset(
            test_smiles,
            enclose=True,
            seq_len=pretrain_config.get("seq_len"),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 1),
        persistent_workers=True,
    )
    test_loader = (
        None
        if test_dataset is None
        else DataLoader(
            test_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config.get("batch_size", 32),
            num_workers=config.get("num_workers", 1),
            persistent_workers=True,
        )
    )
    print(
        f"train dataset size: {len(train_dataset)}, train dataloader size: {len(train_loader)}"
    )

    # copy over any hyperparameters that aren't specified from the pre-training config
    for key, value in pretrain_config.items():
        if key not in config.keys():
            config[key] = value

    if args.arch == "gpt":
        model = GPTLightning.load_from_checkpoint(
            os.path.join(config.get("pretrained_model_path"), "model.ckpt"),
            config=config,
            map_location=torch.device("cpu"),
        )
    elif args.arch == "rnn":
        model = CharRNNLightning.load_from_checkpoint(
            os.path.join(config.get("pretrained_model_path"), "model.ckpt"),
            config=config,
            map_location=torch.device("cpu"),
        )
    else:
        raise ValueError(f"Unrecognized model {args.arch}")

    with open(os.path.join(config.get("model_path"), "config.json"), "w") as fp:
        json.dump(config, fp)

    model, logger = train(model, config, train_loader, test_loader)
