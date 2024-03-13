import torch
from einops import rearrange

def get_mask(x: torch.Tensor, mask_prob: float = 0.08, mask_length: int = 10, mask_pad: int = 0) -> torch.Tensor:
    """
    mask
    - 8% frames are randomly selected as the start of the mask
    - 10 consecutive frames are masked
    """
    B, T, C = x.size()
    nt = T // mask_length
    mask_ones = torch.ones(B, nt, mask_length, C, device=x.device, dtype=x.dtype)
    # Generate random probabilities for each frame
    rand_probs = torch.rand(B, nt, device=x.device)
    # Create a mask based on the specified probability
    mask = (rand_probs < mask_prob).unsqueeze(-1).unsqueeze(-1)
    mask_ones.masked_fill_(mask, mask_pad)
    mask_ones = rearrange(mask_ones, 'b nt ts c -> b (nt ts) c')
    # pad the mask_ones Tensor to the same length as x
    mask_pad = torch.zeros(B, T - mask_ones.size(1), C, device=x.device, dtype=x.dtype)
    mask_ones = torch.cat([mask_ones, mask_pad], dim=1)
    x = x * mask_ones
    return x

# B = 32
# T = 1200
# C = 64
# x = torch.rand(B, T, C)

# mask_prob = 0.08
# mask_length = 10
# mask_pad = 0

# # usage example
# masked_x = get_mask(x, mask_prob, mask_length, mask_pad)
# print(masked_x.shape)
# print(masked_x[0, :200, :2])