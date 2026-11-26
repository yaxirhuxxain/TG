import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from models.ActiveTrans_master import EmbeddingBlock, ActiveTransLayer, ClassificationBlock


def stack_scores(scores):
    stacked_list = {}
    for out in scores:
        for key, val in out.items():
            if key in stacked_list:
                stacked_list[key].append(val)
            else:
                stacked_list[key] = [val]

    _template = {}
    for key, val in stacked_list.items():
        _template[key] = torch.stack(val).mean()

    return _template


class ActiveTransModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size=300,
            ff_size=1200,
            max_pos_embed=4096,
            n_layers=6,
            n_heads=6,
            layer_norm_eps=1e-12,
            bias=False
    ):
        super(ActiveTransModel, self).__init__()

        self.embeddings = EmbeddingBlock(
            vocab_size, hidden_size, max_pos_embed)

        self.layer = nn.ModuleList([
            ActiveTransLayer(ff_size, hidden_size,
                             n_heads, layer_norm_eps, bias)
            for _ in range(n_layers)
        ])
        self.classifier = ClassificationBlock(
            embed_weights=self.embeddings.words.weight, bias=bias)

        self.loss_fn = F.cross_entropy

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states, attention_mask=attention_mask)

        hidden_states = hidden_states[:, -1]
        return self.classifier(hidden_states)


class ActiveTrans(pl.LightningModule):

    def __init__(self,
                 n_layer,
                 n_embd,
                 ff_dim,
                 d_kv,
                 n_head,
                 vocab_size,
                 optimizer,
                 optimizer_lr,
                 loss_fn,
                 ):

        super(ActiveTrans, self).__init__()

        # to save train and validation scores
        self.train_scores = []
        self.val_scores = []

        self.vocab_size = vocab_size
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        if loss_fn == "cross_entropy":
            self.loss_fn = F.cross_entropy
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

        self.transformer = ActiveTransModel(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            ff_size=ff_dim,
            max_pos_embed=4096,
            n_layers=n_layer,
            n_heads=n_head,
            layer_norm_eps=1e-12,
            bias=False
        )

    def forward(self, x):
        # attention masking reduce the modeling performance
        logits = self.transformer(input_ids=x, attention_mask=None)
        return logits

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
        elif self.optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.optimizer_lr)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # self == model
        loss = self.loss_fn(logits, y)
        _scores = {'loss': loss}
        self.log_dict(_scores)  # logging is important for tensorboard
        return _scores

    def training_epoch_end(self, outputs):
        _scores = stack_scores(outputs)
        train_scores = _scores  # train_scores local variable as dict
        train_scores['current_epoch'] = float(self.current_epoch)
        # self.train_scores global variable as a list
        self.train_scores.append(train_scores)
        self.log_dict(_scores)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)  # self == model
        loss = self.loss_fn(logits, y)
        _scores = {'val_loss': loss}
        self.log_dict(_scores)  # logging is important for tensorboard
        return _scores

    def validation_epoch_end(self, outputs):
        _scores = stack_scores(outputs)
        val_scores = _scores  # val_scores local variable as dict
        val_scores['current_epoch'] = float(self.current_epoch)
        # self.val_scores global variable as a list
        self.val_scores.append(val_scores)
        self.log_dict(_scores)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # self == model
        loss = self.loss_fn(logits, y)
        _scores = {'test_loss': loss}
        self.log_dict(_scores)  # logging is important for tensorboard
        return _scores

    def test_epoch_end(self, outputs):
        _scores = stack_scores(outputs)
        self.log_dict(_scores)

    def on_train_epoch_end(self):
        print('\n')

    def on_validation_epoch_end(self):
        print('\n')

    def on_test_epoch_end(self):
        print('\n')
