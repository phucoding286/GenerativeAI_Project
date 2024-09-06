import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __transformer import Transformer
from __positionalEmbedding import PositionalEmbedding
from __masking import create_key_mask, create_key_padding_mask
from __training import trainer

import numpy as np
from torch import nn
from torch import Tensor
from typing import Union, Callable, Any
from torch.nn import functional as F



class ModelT1(nn.Module):

    def __init__(self, sequence_length: int, vocab_size: int, embedding_dim: int = 512,
                embed_dropout: float = 0.1, padding_idx: int = 0, d_model: int = 512,
                num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                dim_feedforward: int = 2048, transformer_dropout: int = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 0.00001, bias: bool = True, device: Any | None = None,
                dtype: Any | None = None):
        super().__init__()
        self.position_embedding = PositionalEmbedding(
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout=embed_dropout,
            padding_idx=padding_idx,
            device=device
        )
        self.transformer = Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.linear_out = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    
    def forward(self, x, y):
        src_mask = create_key_mask(sz=x.shape[1], device=device)
        tgt_mask = create_key_mask(sz=y.shape[1], device=device)
        src_key_padding_mask = create_key_padding_mask(x, device)
        tgt_key_padding_mask = create_key_padding_mask(y, device)
        x = self.position_embedding(x)
        y = self.position_embedding(y)
        x = self.transformer(
            src=x, tgt=y,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.linear_out(x)
    


from tokenizer_ml import TokenizerML

with open("./model-t1 (torch) (transformer) (deverloping)/source_sequences.txt", "r",\
encoding="utf8") as f: x_train = f.read().splitlines()[:10]
with open("./model-t1 (torch) (transformer) (deverloping)/target_sequences.txt", "r",\
encoding="utf8") as f: y_train = f.read().splitlines()[:10]

tokenizer_model = TokenizerML(
    x_train,
    y_train,
    max_len_char=5,
    padding=50,
    num_models=3
)

model = ModelT1(
    sequence_length=tokenizer_model.padding,
    vocab_size=len(tokenizer_model.vocabuliary_dict_to_idx),
    embedding_dim=512,
    embed_dropout=0.1,
    padding_idx=0,
    d_model=512,
    num_heads=16,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=700,
    transformer_dropout=0.1,
)

x = torch.tensor(tokenizer_model.x_sequences_batch)
y = torch.tensor(tokenizer_model.y_sequences_batch)

# model.load_state_dict(torch.load("./model-t1 (torch) (transformer) (deverloping)/model_t1.pt"))
trainer(
    x,
    y,
    model,
    vocab_size=len(tokenizer_model.vocabuliary_dict_to_idx),
    epochs=1,
    batch_size=16,
    lr=0.001,
    device=device,
    output_testing=True,
    tokenizer_model=tokenizer_model
)
torch.save(model.state_dict(), "./model-t1 (torch) (transformer) (deverloping)/model_t1.pt")

x_test = torch.tensor(tokenizer_model.encode_input(["hi"])).long()
y_test = torch.tensor([[1] + [0 for _ in range(x_test.shape[1]-1)]]).long()

output = torch.argmax(model(x_test, y_test), -1)
print(tokenizer_model.decode_output(output))