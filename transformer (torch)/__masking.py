from typing import Optional
import torch
from torch import Tensor


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