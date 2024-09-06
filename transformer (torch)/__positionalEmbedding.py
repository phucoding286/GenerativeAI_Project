import torch
from torch import nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self, device):
        even_i = torch.arange(0, self.d_model, 2, device=device).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length, device=device).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim, dropout=0.1, padding_idx=0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.positional_encoding_dropout = nn.Dropout(p=dropout)
        self.position = PositionalEncoding(embedding_dim, sequence_length)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=self.device)
    

    def forward(self, x):
        x = self.embedding(x)
        position = self.position.forward(self.device)
        out = self.positional_encoding_dropout(x + position)
        return out.to(self.device)