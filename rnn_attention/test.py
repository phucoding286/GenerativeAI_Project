import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __rnnAttention import RNN
from __training import trainer

import numpy as np


x = torch.tensor(np.random.randint(1, 50, size=(10, 50))).to(device)
y = torch.tensor(np.random.randint(1, 50, size=(10, 50))).to(device)
padding = torch.zeros((10, 5)).to(device)
x = torch.concatenate([x, padding], -1).long().to(device)
y = torch.concatenate([y, padding], -1).long().to(device)

model = RNN(
    vocab_size=51,
    embedding_dim=1024,
    input_dim=1024,
    output_dim=1024,
    padding_idx=0,
    encoder_num_layers=2,
    decoder_num_layers=2,
    batch_first=True,
    dropout=0.1,
    device=device,
)
trainer(x, y, model, vocab_size=51, epochs=20, batch_size=16, device=device, lr=0.001)