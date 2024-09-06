import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
    

    def forward(self, q, k):
        W1 = self.W1(q).to(self.device)
        W2 = self.W2(k).to(self.device)
        W2 = W2.unsqueeze(1)
        scores = torch.tanh(W1 + W2).to(self.device)
        scores = self.V(scores).to(self.device)
        weights = torch.softmax(scores, dim=-1)
        values = (weights * q).to(self.device)
        values = torch.sum(values, dim=1)
        return values.to(self.device), weights.to(self.device)