from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import torch
from torch import nn
from typing import Optional, Callable



def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def create_key_mask(
        sz: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
    """Creates a key mask (causal mask) with True for masked positions."""
    if device is None:
        device = torch.device('cpu')
    masks = _generate_square_subsequent_mask(sz)
    boolean_mask = (masks == -torch.inf)
    return boolean_mask.to(device)


def create_key_padding_mask(
        inputs: Tensor,
        device: Optional[torch.device] = None,
    ) -> Tensor:
    """Creates a padding mask where input tokens equal to 0 are masked."""
    if device is None:
        device = torch.device('cpu')
    key_padding_mask = (inputs == 0)
    return key_padding_mask.to(device)



def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")





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






class AutoRegressiveLayer(torch.nn.Module):

    def __init__(self, d_model: int = 512, nhead=8,
                  dim_ffn: int = 1024, dropout: float = 0.1, activation = F.gelu,
                  layer_norm_eps: float = 0.00001, batch_first: bool = True,
                  norm_first: bool = True, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)

        self.linear1 = torch.nn.Linear(d_model, dim_ffn, bias=bias, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_ffn, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def forward(
            self,
            inp: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = True) -> torch.Tensor:

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=inp.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=inp.dtype,
            check_other=False,
        )

        x = inp
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), mask, key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, mask, key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x
    




class ModelG1(torch.nn.Module):

    def __init__(self, d_model: int = 512, max_sequence_length: int = 200, padding_idx=0,
                  vocab_size: int = None, embed_dropout: float = 0.001, nhead=8, num_layers=6,
                  dim_ffn: int = 1024, dropout: float = 0.1, activation = F.gelu,
                  layer_norm_eps: float = 0.00001, batch_first: bool = True,
                  norm_first: bool = True, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()

        self.device = device

        self.pos_embed = PositionalEmbedding(max_sequence_length, vocab_size, d_model,
                                             embed_dropout, padding_idx, device)
        self.auto_regressive = nn.ModuleList([
            AutoRegressiveLayer(d_model, nhead, dim_ffn, dropout, activation,
                                layer_norm_eps, batch_first, norm_first,
                                bias, device, dtype) for _ in range(num_layers)])
        self.linear_out = torch.nn.Linear(d_model, vocab_size, bias, device, dtype)
    
    def forward(self, x):
        mask = create_key_mask(sz=x.shape[-1], device=self.device)
        padding_mask = create_key_padding_mask(x, device=self.device)
        x = self.pos_embed(x)
        for layer in self.auto_regressive: x = layer(x, mask, padding_mask)
        x = self.linear_out(x)
        return x