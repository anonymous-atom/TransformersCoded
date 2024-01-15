import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """
    A class used to represent the input embeddings for a model.

    ...

    Attributes
    ----------
    d_model : int
        The dimension of the model
    vocab_size : int
        The size of the vocabulary
    embedding : nn.Embedding
        The embedding layer of the model

    Methods
    -------
    __init__(self, d_model: int, vocab_size: int)
        Initializes the InputEmbeddings with the dimension of the model and the size of the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        # Create a vector of shape (seq_length, 1) for each word
        position = torch.arange(0, seq_length - 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even pos.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add Batch dimension
        pe = pe.unsqueeze(0)  # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __int__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__int__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Bath, seq_len, d_model) ---> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_length, d_k) --> (Batch, h, seq_length, seq_length)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_length, seq_length)
        # Because we multiplied query and key (Batch, h, seq_length, d_k) * (Batch, h, d_k, seq_length)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores  # (Batch, h, seq_length, d_k)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # ( Batch , seq_length, d_model )
        key = self.w_k(k)
        values = self.w_v(v)

        # (Batch, seq_length, d_model) --> (Batch, seq_length, h, d_k) --> (Batch, h, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = values.view(values.shape[0], values.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attentions_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_length, d_k) --> (Batch, seq_length, h, d_k) --> (Batch, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_length, d_model) --> (Batch, seq_length, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __int__(self, dropout: float):
        super().__int__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __int__(self, layers: nn.ModuleList):
        super().__int__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
