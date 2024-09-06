from __gpt import GPT
from __training import trainer
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = torch.tensor(np.random.randint(low=1, high=20, size=(100, 20)))
y_train = torch.tensor(np.random.randint(low=1, high=20, size=(100, 20)))
x_train = torch.concatenate([x_train, torch.zeros((x_train.shape[0], 5))], dim=-1)
y_train = torch.concatenate([y_train, torch.zeros((x_train.shape[0], 5))], dim=-1)

model = GPT(vocab_size=21, sequence_length=x_train.shape[1], num_layers=2, d_model=1024, dim_feedforward=2048, device=device)
trainer(x_train, y_train, model, vocab_size=21, epochs=100, batch_size=16, lr=0.001)