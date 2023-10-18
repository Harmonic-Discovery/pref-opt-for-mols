import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
import neptune
from sklearn.model_selection import train_test_split
from neptune_pytorch import NeptuneLogger
from pref_opt_for_mols.models import GPT, CharRNN, DPO
from pref_opt_for_mols.dataset import PreferencePairDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    required=True,
    type=str,
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="if true, turn off neptune logging",
)
parser.add_argument(
    "--name", type=str, help="experiment name (default untitled)", default="Untitled"
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", default=None, type=int)

args = parser.parse_args()


def getset(key, val=None):
    try:
        return config[key]
    except KeyError:
        config[key] = val
        return val


def get_dataloader(dataset, key, verbose=True):
    if dataset is None:
        if verbose:
            print(f"no {key} dataset provided")
        return

    loader = dataset.dataloader(
        shuffle=(key == "train"),
        pin_memory=True,
        batch_size=getset("batch_size", 32),
        num_workers=getset("num_workers", 4),
    )
    if verbose:
        print(
            f"{key} dataset size: {len(dataset)}, "
            f"{key} dataloader size: {len(loader)}"
        )
    return loader


def get_datasets(tokenizer):
    val_split = getset("val_split", 0.0)

    n_train = getset("n_train", "auto")
    if n_train == "auto":
        n_train = len(pd.read_csv(config["train_smiles_path"]))
        n_train = min(n_train, 200000)

    data = pd.read_csv(config["train_smiles_path"])
    train_data, val_data = train_test_split(data, test_size=val_split)

    common = dict(
        smiles_key="smiles",
        label_key="label",
        seq_len=getset("seq_len", 512),
        tokenizer=tokenizer,
        enclose=True,
    )

    train_dataset = PreferencePairDataset(
        train_data,
        n=n_train,
        aug_prob=getset("aug_prob", 0.0),
        **common,
    )

    val_dataset = None
    if val_split > 0.0:
        val_dataset = PreferencePairDataset(
            val_data,
            n=int(n_train * val_split),
            aug_prob=0.0,
            **common,
        )

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Parse config and set seed
    torch.set_float32_matmul_precision("high")  # NVIDIA A5000

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    with open(args.config) as fp:
        config = json.load(fp)

    # Model loading
    if config["arch"] == "gpt":
        model_cls = GPT
    elif config["arch"] == "rnn":
        model_cls = CharRNN
    else:
        raise ValueError(f"Unrecognized model '{args.arch}'")

    ckpt_path = os.path.join(config["reference_path"], "model.ckpt")
    with open(os.path.join(config["reference_path"], "config.json")) as f:
        ref_config = json.load(f)

    reference = model_cls.load_from_checkpoint(
        ref_config,
        ckpt_path,
        disable_dropout=False,
        device="cpu",
    )
    policy = model_cls.load_from_checkpoint(
        ref_config, ckpt_path, disable_dropout=False, device="cpu"
    )
    print("Loaded reference and policy models")

    # Make train, val, and test datasets
    train_dataset, val_dataset = get_datasets(policy.tokenizer)
    train_loader = get_dataloader(train_dataset, "train")
    val_loader = get_dataloader(val_dataset, "val")

    # Training and saving
    if config.get("neptune_project"):
        npt_run = neptune.init_run(
            project=config.get("neptune_project"),
            api_token=os.environ.get("NEPTUNE_API_KEY"),
            mode="debug" if args.debug else "async",
            name=args.name,
        )
        npt_logger = NeptuneLogger(run=npt_run, model=policy)
    else:
        npt_logger = None
    trainer = DPO(
        reference, policy, config, logger=npt_logger, run=npt_logger, device=args.device
    )
    trainer.save_configs(config["model_path"])
    trainer.train(train_loader, val_loader)
