# *** Basic Imports *** #
import argparse
import os
import pickle
import warnings

import numpy as np
import torch
import yaml
from torch.multiprocessing import Pool
from tqdm import tqdm

from metrics import bleu_score, word_levenshtein, stack_scores
from models.ActiveTrans_pl import ActiveTrans as MODEL
from models.dataset import incremental_learning

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


# *** Main Evaluator Configurations *** #
class Evaluator(object):

    def __init__(
            self,
            model_name,
            model_folder,
            data_folder,
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
            n_workers=1,
    ):

        self.init_epoch = None
        self.in_dir = None  # it will be set by each fold
        self.word_idx = None
        self.model_folder = model_folder
        self.data_folder = data_folder
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
        self.context_size = context_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.val_interval = val_interval
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.n_workers = n_workers

    def encoder(self, item):
        return [self.word_idx[word] if word in self.word_idx else self.word_idx['<idf>'] for word in item.split()]

    def prepare_datasets(self, data_files, max_samples=None):

        encoded_examples = [self.encoder(item.replace('\n', ''))
                            for item in incremental_learning(data_files)]

        if max_samples is not None:
            encoded_examples = encoded_examples[0:max_samples]

        return encoded_examples

    def test_model(self, ctx=1):

        torch.cuda.empty_cache()

        print("Model: {}".format(self.model_name))
        self.in_dir = os.path.join(self.model_folder, f"{self.model_name}")

        if not os.path.exists(self.in_dir):
            print(f"{self.model_name} | Model is not trained...\n\n")

        print("\t> Loading Model & Preparing data")

        if not os.path.exists(os.path.join(self.in_dir, "model.ckpt")):
            print(f"{self.model_name} | Model is not trained...\n\n")
            return 0
        else:
            model = MODEL.load_from_checkpoint(
                checkpoint_path=os.path.join(self.in_dir, "model.ckpt"),
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

        with open(os.path.join(self.in_dir, "word_idx.pkl"), 'rb') as rFile:
            self.word_idx = pickle.load(rFile)
            self.idx_word = {v: k for k, v in self.word_idx.items()}
            rFile.close()

        data_files = {"test": [os.path.join(self.data_folder, "test.txt")]}

        test_dataset = self.prepare_datasets(data_files["test"], max_samples=None)
        test_dataset_chunked = np.array_split(np.array(test_dataset), self.n_batch)

        print("\t> Model Testing")
        with torch.no_grad():

            pool_jobs = []
            pbar = tqdm(total=len(test_dataset_chunked))

            def pbar_update(_):
                pbar.update()

            with Pool(processes=self.n_workers if self.n_workers > 0 else 1) as pool:
                for batch in test_dataset_chunked:
                    p = pool.apply_async(
                        gen_mp,
                        args=(
                            model.eval().to(device), batch,
                            ctx, self.word_idx, self.context_size,
                        ),
                        callback=pbar_update
                    )
                    pool_jobs.append(p)

                pool.close()
                pool.join()

                scores = [p.get() for p in pool_jobs]  # collect job results

                scores = stack_scores(scores)
                print(scores)
                pbar.close()
                torch.cuda.empty_cache()

        print(f">>> {self.model_name} | Model testing is done...\n\n")


def gen_mp(model, batch, ctx=1, word_idx=None, context_size=10):
    X_true = []
    X_pred = []
    bleu = []
    levenshtein = []

    idx_word = {val: key for key, val in word_idx.items()}
    break_list = [word_idx[";"], word_idx["{"], word_idx["}"]]
    for row in batch:
        X_true.append([idx_word[idx] for idx in row])  # True sequence
        X_pred.append([idx_word[idx] for idx in row[0:ctx]])  # initialize with seed tokens

    # Finding padding length and preparing seed tokens
    bs = len(batch)
    init_tokens = torch.as_tensor([item[0:ctx] for item in batch])  # getting init words for seed
    pad_len = context_size - init_tokens.size(1)  # pad length
    padding_mtx = torch.ones((bs, pad_len)) * word_idx['<pad>']  # padding matrix
    seed = torch.cat((padding_mtx, init_tokens), 1).type(torch.long)  # seed matrix padded
    seed = seed.to(device)

    batch_pred = None
    while True:
        logits = model(seed)
        sampled_token_idx = logits.topk(k=1, dim=1).indices
        seed = torch.cat((seed[:, 1:], sampled_token_idx), 1)  # adding pred and sliding context

        if batch_pred is None:
            batch_pred = sampled_token_idx
        else:
            batch_pred = torch.cat((batch_pred, sampled_token_idx), 1)

        if batch_pred.size(1) > context_size - 1:
            break

    for idx, row in enumerate(batch_pred.cpu().numpy()):
        utterance = []
        for i in list(row):
            utterance.append(i)
            if i in break_list:
                break

        X_pred[idx] = X_pred[idx] + [idx_word[ut] for ut in utterance]
        bleu.append(bleu_score(preds=" ".join(X_pred[idx]), target=[" ".join(X_true[idx])], n_gram=4, smooth=True))
        levenshtein.append(word_levenshtein(preds=X_pred[idx], target=X_true[idx]))

    return {
        "levenshtein": torch.scalar_tensor(np.mean(levenshtein)),
        "BLEU4": torch.scalar_tensor(np.mean(bleu))
    }


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser(description='Model Evaluator')
    parser.add_argument('-d', '--data_folder', type=str, default="data/splits/java", help='input folder location')
    parser.add_argument('-i', '--model_folder', type=str, default="saveModels/java", help='output folder location')
    parser.add_argument('-m', '--model', type=str, default="ActiveTrans", help='ActiveTrans | ActiveTransPlus')

    # Setting Prams
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open('models/configs.yml', 'r') as file:
        configs = yaml.safe_load(file)

    Evaluator(
        data_folder=args.data_folder,
        model_folder=args.model_folder,
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
        n_drop=configs['Shared']['n_drop'],
        n_batch=configs['Shared']['translation_batch_size']
    ).test_model()


if __name__ == "__main__":
    main()
