import argparse
import pandas as pd
from pref_opt_for_mols.models import GPTLightning, CharRNNLightning
from pref_opt_for_mols.dataset import NextTokenSmilesDataset, SmilesVocabulary, SmilesTokenizer
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


def load_moses(split="train"):
    assert split in ["train", "test"]
    return pd.read_csv(f"data/{split}.csv")["smiles"].tolist()


if __name__ == "__main__":
    with open(args.config) as fp:
        config = json.load(fp)

    assert config.get("model_path") is not None

    os.makedirs(config.get("model_path"), exist_ok=True)

    train_smiles = load_moses("train")
    test_smiles = load_moses("test")

    # 1. Explicitly create SmilesVocabulary and SmilesTokenizer
    smiles_vocab = SmilesVocabulary() # Uses default symbols
    tokenizer = SmilesTokenizer(vocabulary=smiles_vocab)

    # Update config with vocab_size from the tokenizer
    # This should reflect the actual size after potential HF alignment if tokenizer.hf_tokenizer is used
    # For now, save_pretrained will use smiles_vocab to create files, then load hf_tokenizer.
    # So, the vocab size from smiles_vocab.symbols is the source of truth for GPT2Config.
    config["vocab_size"] = len(smiles_vocab.symbols) # or len(tokenizer.vocabulary.symbols)

    # Ensure model_path exists for saving tokenizer files
    model_output_path = config.get("model_path")
    os.makedirs(model_output_path, exist_ok=True)

    # 2. Save tokenizer files (vocab.json, merges.txt) to model_output_path
    # This allows GPT2Config (inside GPT class) to potentially find them if it looks,
    # and makes them available for other scripts.
    # The SmilesTokenizer.save_pretrained method will create the necessary files
    # from its internal SmilesVocabulary if its hf_tokenizer is not yet set.
    # After this call, tokenizer.hf_tokenizer might be initialized if save was successful.
    tokenizer.save_pretrained(model_output_path)
    print(f"Tokenizer vocabulary saved to {model_output_path}")

    # Add tokenizer path to config for clarity, though it's same as model_path here
    config["tokenizer_path"] = model_output_path 


    # 3. Initialize datasets with the created tokenizer
    train_dataset = NextTokenSmilesDataset(
        smiles_list=train_smiles,
        tokenizer=tokenizer, # Pass the initialized tokenizer
        enclose=True,
        seq_len=config.get("seq_len"),
        aug_prob=config.get("aug_prob", 0.0),
    )

    test_dataset = NextTokenSmilesDataset(
        smiles_list=test_smiles,
        tokenizer=tokenizer, # Pass the initialized tokenizer
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
    print(f"Set config['vocab_size'] = {config['vocab_size']}")


    if args.arch == "gpt":
        # config["vocab_size"] is already set from the tokenizer
        config["block_size"] = config.get("seq_len") - 1 # block_size is n_positions for GPT2
        # GPTLightning will pass this config to GPT, which initializes GPT2Config and GPT2LMHeadModel
        model = GPTLightning(config)
    elif args.arch == "rnn":
        model = CharRNNLightning(config)
    else:
        raise ValueError(f"Unrecognized model {args.arch}")

    # Save the final configuration (including updated vocab_size and tokenizer_path)
    # This should be done AFTER all modifications to config are complete.
    with open(os.path.join(model_output_path, "config.json"), "w") as fp:
        json.dump(config, fp)
    print(f"Final configuration saved to {os.path.join(model_output_path, 'config.json')}")

    model, logger = train(model, config, train_loader, test_loader)
