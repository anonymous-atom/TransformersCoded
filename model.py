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

        # Create a vector of shape (seq_lenght) for each word
        position = torch.arange(0, seq_length - 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        #Apply sin to even pos.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add Batch dimension
        pe = pe.unsqueeze(0)  # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, eps: float == 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * ( x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __int__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__int__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        # (Bath, seq_len, d_model) ---> (Batch, seq_len, d_ff) --> (Batch, deq_model, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
