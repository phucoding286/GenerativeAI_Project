import torch
import torch.nn.functional as F
from __transformerDecoderLayer import TransformerDecoderLayer
from __transformerDecoderLayerSequential import TransformerDecoder
from __masking import create_key_mask, create_key_padding_mask
from __positionalEmbedding import PositionalEmbedding
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPT(torch.nn.Module):

    def __init__(self, vocab_size: int, sequence_length: int, num_layers: int = 2,
        d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
        activation=F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = True,
        norm_first: bool = True, bias: bool = True, device=None, dtype=None,
        **kwargs):
        super(GPT, self).__init__(**kwargs)

        self.device = device
        self.embedding = PositionalEmbedding(
            sequence_length,
            vocab_size,
            d_model,
            dropout,
            padding_idx=0,
            device=device
        )
        self.decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.decoder = TransformerDecoder(
            self.decoder_layer,
            num_layers
        ).to(self.device)
        self.linear_out = torch.nn.Linear(
            d_model,
            vocab_size,
            bias,
            device=device
        )


    def forward(self, x):
        seq_len = x.size(1)
        x = x.long().to(self.device)
        key_mask = create_key_mask(seq_len, device=self.device)
        key_padding_mask = create_key_padding_mask(x, device=self.device)
        x = self.embedding(x).to(self.device)
        x = self.decoder_layer(
            inputs=x,
            inputs_mask=key_mask,
            inputs_key_padding_mask=key_padding_mask,
        ).to(self.device)
        out = self.linear_out(x)
        return out.to(self.device)
    

if __name__ == "__main__":
    # testing
    model = GPT(vocab_size=20, sequence_length=20, num_layers=1, dim_feedforward=1024, device=device)
    x_test = torch.tensor(np.random.uniform(low=0, high=20, size=(10, 20))).int().to(device)
    print(model(x_test))
