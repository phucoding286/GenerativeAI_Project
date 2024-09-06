import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __attention import Attention

from torch import nn 
from typing import Optional
import numpy as np


class RNN(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, input_dim: int, output_dim: int, padding_idx: int = 0,
                encoder_num_layers: int = 6, decoder_num_layers: int = 6, batch_first: bool = False,
                dropout: float = 0.1, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )
        self.rnn_encoder = nn.RNN(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=encoder_num_layers,
            batch_first=batch_first,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        self.rnn_decoder = nn.RNN(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=decoder_num_layers,
            batch_first=batch_first,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        self.attention = Attention(
            input_dim=output_dim,
            hidden_dim=output_dim,
            device=device
        )
        self.linear_out = nn.Linear(
            in_features=input_dim*2,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.rnn_encoder(x)
        encoder_output = x[0].clone().to(self.device)
        x = self.rnn_decoder(x[0], hx=x[1])
        decoder_output = x[0].clone().to(self.device)
        x = self.attention(decoder_output, encoder_output)
        x = torch.cat([decoder_output, x[0]], dim=-1)
        out = self.linear_out(x)
        return out.to(self.device)