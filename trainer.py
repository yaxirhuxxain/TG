# *** Basic Imports *** #
import argparse
import glob
import os
import pickle
import re
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import callbacks
from torch.utils.data import DataLoader

from models.ActiveTrans_pl import ActiveTrans as MODEL
from utils import build_vocab

# *** Basic Configurations *** #
warnings.filterwarnings("ignore")

# Set random values
seed_val = 786
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

torch.backends.cudnn.benchmark = True  # SpeedUP: Turn on cudNN benchmarking
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Enable PyTorch to use the GPU.


# *** Main Trainer Configurations *** #
class Trainer(object):

    def __init__(
            self,
            model_name,
            in_folder,
            out_folder,
            context_size,
            n_layer,
            n_embd,
            n_head,
            ff_dim,
            d_kv,
            vocab_size,
            optimizer,
            optimizer_lr,
            loss_fn,
            epochs,
            warmup_epochs,
            patience,
            val_interval,
            n_batch,
            n_drop,
            n_workers,
    ):

        self.init_epoch = None
        self.out_dir = None  # it will be set by each fold
        self.word_idx = None
        self.model_name = model_name
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.ff_dim = ff_dim
        self.d_kv = d_kv
        self.vocab_size = vocab_size
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.loss_fn = loss_fn
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.context_size = context_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.val_interval = val_interval
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.n_workers = n_workers

    def train_model(self):

        print("Model: {} > Data Folder: {}".format(
            self.model_name, self.in_folder))
        self.out_dir = os.path.join(self.out_folder, f"{self.model_name}")

        print(f"Model Save Folder: {self.out_dir}")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        print("\n*** Building Model & Trainer ***")

        print("\t > Preparing Trainer")
        earlyStopCallBack = callbacks.early_stopping.EarlyStopping(
            patience=self.patience,
            monitor='val_loss',
            mode='min'
        )

        trainer = pl.Trainer(
            gpus=1,
            strategy='dp',
            min_epochs=self.warmup_epochs,
            max_epochs=self.epochs,
            callbacks=[earlyStopCallBack],
            val_check_interval=self.val_interval,  # check every n times of an epoch
        )

        model = None
        if os.path.exists(self.out_dir):
            if os.path.exists(os.path.join(self.out_dir, "model.ckpt")):
                print(f"{self.model_name} | Model is already trained...\n\n")
                return 0
            else:

                last = None
                checkpoints = glob.glob(os.path.join(self.out_dir, 'weights-improvement-*.ckpt'))
                if checkpoints:
                    last = max(checkpoints)
                    self.init_epoch = int(re.findall("\d+", last.replace(self.out_dir, ""))[0])
                    print(
                        f"\t>Check point found resume training from: Epoch {self.init_epoch + 1}")
                    trainer.fit_loop.current_epoch = self.init_epoch + 1
                    model = MODEL.load_from_checkpoint(
                        checkpoint_path=last,
                        n_layer=self.n_layer,
                        n_embd=self.n_embd,
                        ff_dim=self.ff_dim,
                        d_kv=self.d_kv,
                        n_head=self.n_head,
                        vocab_size=self.vocab_size,
                        optimizer=self.optimizer,
                        optimizer_lr=self.optimizer_lr,
                        loss_fn=self.loss_fn,
                    )
                else:
                    model = MODEL(
                        n_layer=self.n_layer,
                        n_embd=self.n_embd,
                        ff_dim=self.ff_dim,
                        d_kv=self.d_kv,
                        n_head=self.n_head,
                        vocab_size=self.vocab_size,
                        optimizer=self.optimizer,
                        optimizer_lr=self.optimizer_lr,
                        loss_fn=self.loss_fn,
                    )

        print("\n*** Model Summary ***")
        print(model)

        print('\n*** Preparing data ***')

        data_files = {"train": [os.path.join(self.in_folder, "train.txt")],
                      "valid": [os.path.join(self.in_folder, "dev.txt")]}

        print("\t> Abstracting Fine-Grained Tokens")
        self.word_idx, _, _ = build_vocab(
            files=data_files['train'],
            vocab_size=self.vocab_size,
            tokenizer_fn=None
        )

        self.vocab_size = len(self.word_idx)

        from models.dataset import Dataset

        print(
            f'\t> Preparing data with {"Incremental" if self.model_name == "ActiveTransPlus" else "WindowWise"} strategy')
        train_dataset = Dataset(
            data_files['train'],
            context_size=self.context_size,
            word_idx=self.word_idx,
            type="Incremental" if self.model_name == "ActiveTransPlus" else "WindowWise",
            num_samples=None
        )

        valid_dataset = Dataset(
            data_files['valid'],
            context_size=self.context_size,
            word_idx=self.word_idx,
            type="Incremental" if self.model_name == "ActiveTransPlus" else "WindowWise",
            num_samples=None
        )

        self.train_rows = len(train_dataset)
        self.valid_rows = len(valid_dataset)
        print(f"\t > Train Examples: {self.train_rows}")
        print(f"\t > Validation Examples: {self.valid_rows}")

        # DataLoader will only pass one batch size of examples to collector function
        def collector(examples):
            examples = torch.tensor(examples, dtype=torch.int64)
            return examples[:, :-1], examples[:, -1]

        train_loader = DataLoader(
            train_dataset,
            collate_fn=collector,
            batch_size=self.n_batch,
            num_workers=self.n_workers,
            persistent_workers=True if self.n_workers > 0 else False,
            pin_memory=True,
            shuffle=False
        )
        valid_loader = DataLoader(
            valid_dataset,
            collate_fn=collector,
            batch_size=self.n_batch,
            num_workers=self.n_workers,
            persistent_workers=True if self.n_workers > 0 else False,
            pin_memory=True,
            shuffle=False
        )

        print("\n*** Model training ***")
        trainer.fit(model, train_loader, valid_loader)

        print("\n*** Saving Model ***")
        self.model_save(trainer)
        torch.cuda.empty_cache()

        print(f">>> {self.model_name} | Model training is done...\n\n")

    def model_save(self, trainer):

        # Saving model
        trainer.save_checkpoint(os.path.join(self.out_dir, "model.ckpt"))

        # Saving vocab
        with open(os.path.join(self.out_dir, "word_idx.pkl"), "wb") as file:
            pickle.dump(self.word_idx, file)


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('-i', '--in_folder', type=str, default="data/splits/java", help='input folder location')
    parser.add_argument('-o', '--out_folder', type=str, default="saveModels/java", help='output folder location')
    parser.add_argument('-m', '--model', type=str, default="ActiveTrans", help='ActiveTrans | ActiveTransPlus')

    # Setting Prams
    args = parser.parse_args()

    with open('models/configs.yml', 'r') as file:
        configs = yaml.safe_load(file)

    Trainer(
        in_folder=args.in_folder,
        out_folder=args.out_folder,
        model_name=args.model,
        vocab_size=configs['Shared']['vocab_size'],
        context_size=configs['Shared']['context_size'],
        n_layer=configs['Shared']['layers'],
        n_embd=configs['Shared']['embed_dim'],
        n_head=configs['Shared']['heads'],
        ff_dim=configs['Shared']["ff_dim"],
        d_kv=configs['Shared']["d_kv"],
        optimizer=configs['Shared']['optimizer'],
        optimizer_lr=configs['Shared']['optimizer_learn_rate'],
        loss_fn=configs['Shared']['loss_fn'],
        epochs=configs['Shared']['epochs'],
        warmup_epochs=configs['Shared']['warmup_epochs'],
        patience=configs['Shared']['patience'],
        val_interval=configs['Shared']['val_interval'],
        n_batch=configs['Shared']['plus_batch_size'],
        n_drop=configs['Shared']['n_drop'],
        n_workers=configs['Shared']['n_workers']
    ).train_model()


if __name__ == "__main__":
    main()
