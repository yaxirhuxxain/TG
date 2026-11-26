import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# Defining CPU GPU Devices.
GPU = torch.device('cuda')
CPU = torch.device('cpu')


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class PositionEmbedding(nn.Module):
    """Constructs position embeddings of shape `[max_pos_embed, hidden_size]`."""

    def __init__(self, hidden_size=300, max_pos_embed=1200, n_drop=0.05):
        super().__init__()
        self.n_drop = n_drop
        self.embedding = nn.Embedding(max_pos_embed, hidden_size)

    def forward(self, pos_ids):
        position_embeddings = self.embedding(pos_ids)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.n_drop)
        return position_embeddings


class EmbeddingBlock(nn.Module):
    """Construct embeddings from words and positions."""

    def __init__(self, vocab_size, hidden_size=300, max_pos_embed=1200):
        super().__init__()
        self.max_pos_embed = max_pos_embed

        self.words = nn.Embedding(vocab_size, hidden_size)
        self.positions = PositionEmbedding(hidden_size, max_pos_embed)

    def forward(self, input_ids=None, start_idx=0):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        pos_ids = torch.arange(start_idx, start_idx +
                               seq_length, dtype=torch.long, device=GPU)
        pos_ids = pos_ids.unsqueeze(0).expand(input_shape)

        words_embedding = self.words(input_ids)

        if pos_ids.shape[-1] > self.max_pos_embed:
            raise ValueError(
                f"seq_length: {pos_ids.shape[-1]} can not be grater than "
                f"max_pos_embed: {self.self.max_pos_embed}."
            )

        # add positional embeddings
        position_embeddings = self.positions(pos_ids)
        embeddings = words_embedding + position_embeddings
        return embeddings


class ActivatedAttention(nn.Module):
    def __init__(self, hidden_size=300, n_heads=6, bias=False):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) should be multiple of "
                f"the number of attention heads ({n_heads})"
            )

        self.n_heads = n_heads
        self.head_size = int(hidden_size / n_heads)

        self.query = nn.Linear(
            hidden_size, self.n_heads * self.head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.n_heads *
                             self.head_size, bias=bias)
        self.value = nn.Linear(
            hidden_size, self.n_heads * self.head_size, bias=bias)

        self.update = nn.Linear(
            hidden_size, self.n_heads * self.head_size, bias=bias)

    def expand_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, n_heads, seq_length, head_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        query_layer = self.expand_heads(self.query(hidden_states))
        key_layer = self.expand_heads(self.key(hidden_states))
        value_layer = self.expand_heads(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
                           math.sqrt(self.head_size)  # scaling improve performance

        # Apply the attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Activated Update
        update_gate = torch.sigmoid(
            self.expand_heads(self.update(hidden_states)))

        new_gate = torch.matmul(
            attention_scores, key_layer / math.sqrt(self.head_size))

        context_layer = update_gate * value_layer + \
                        (1 - update_gate) * new_gate

        # combining heads for final output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        final_shape = context_layer.size(
        )[:-2] + (self.n_heads * self.head_size,)
        context_layer = context_layer.view(final_shape)

        return context_layer


class FeedForwardNet(nn.Module):
    def __init__(self, ff_size=1200, hidden_size=300, layer_norm_eps=1e-12, bias=False):
        super().__init__()

        self.h2f = nn.Linear(hidden_size, ff_size, bias=bias)
        self.act = QuickGELUActivation()  # gelu based activations works better
        self.f2o = nn.Linear(ff_size, hidden_size, bias=bias)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ff_hidden_states = self.h2f(hidden_states)
        ff_hidden_states = self.act(ff_hidden_states)
        ff_hidden_states = self.f2o(ff_hidden_states) + hidden_states
        hidden_states = self.LayerNorm(
            ff_hidden_states + self.dense(hidden_states) + hidden_states)
        return hidden_states


class ActiveTransLayer(nn.Module):
    def __init__(self, ff_size=1200, hidden_size=300, n_heads=6, layer_norm_eps=1e-12, bias=False):
        super(ActiveTransLayer, self).__init__()
        self.attention = ActivatedAttention(
            hidden_size=hidden_size, n_heads=n_heads, bias=bias)
        self.ff_net = FeedForwardNet(ff_size=ff_size, hidden_size=hidden_size,
                                     layer_norm_eps=layer_norm_eps, bias=bias)

    def forward(self, input_emb, attention_mask=None):
        return self.ff_net(self.attention(input_emb, attention_mask=attention_mask))


class ClassificationBlock(nn.Module):
    def __init__(self, embed_weights, bias=False):
        super(ClassificationBlock, self).__init__()
        embed_shape = embed_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=bias)
        self.decoder.weight = embed_weights  # Tied weight

    def forward(self, hidden_state):
        return self.decoder(hidden_state)
