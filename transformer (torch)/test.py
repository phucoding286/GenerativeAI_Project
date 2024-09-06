import torch
import numpy as np
from __transformer import Transformer
from __positionalEmbedding import PositionalEmbedding
from __masking import create_key_mask, create_key_padding_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestTransformer(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.position_embedding = PositionalEmbedding(25, 21, 512, device=device)
        self.transformer = Transformer(d_model=512, nhead=16, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, device=device)
        self.linear_out = torch.nn.Linear(512, 20, device=device)
    
    def forward(self, x, y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        x = self.position_embedding(x)
        y = self.position_embedding(y)
        x = self.transformer(src=x, tgt=y, src_mask=src_mask, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.linear_out(x)

x = torch.tensor(np.random.randint(1, 20, size=(10, 20))).to(device)
y = torch.tensor(np.random.randint(1, 20, size=(10, 20))).to(device)
padding = torch.zeros((10, 5)).to(device)
x = torch.concatenate([x, padding], -1).long().to(device)
y = torch.concatenate([y, padding], -1).long().to(device)

src_mask = create_key_mask(sz=25, device=device)
tgt_mask = create_key_mask(sz=25, device=device)
src_key_padding_mask = create_key_padding_mask(x, device)
tgt_key_padding_mask = create_key_padding_mask(y, device)

model = TestTransformer(device)
output = model(x, y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

print(output)