import itertools as itt
from functools import lru_cache

import torch
from einops import rearrange



def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    windows = rearrange(x, 'b (nw_t w_t) (nw_h w_h) (nw_w w_w) c -> (b nw_t nw_h nw_w) (w_t w_h w_w) c', 
                        w_t=window_size[0], w_h=window_size[1], w_w=window_size[2])
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = rearrange(windows, '(b nw_t nw_h nw_w) w_t w_h w_w c -> b (nw_t w_t) (nw_h w_h) (nw_w w_w) c',
                  nw_t = D // window_size[0], nw_h = H // window_size[1], nw_w = W // window_size[2])
    return x


@lru_cache
def construct_position_index(window_size):
    """
    Construct the position index to reuse symmetrical parameters of the position bias.

    Args:
    - window_size: A tuple representing the size of the window (pressure level, latitude, longitude).

    Returns:
    - position_index: A flattened tensor representing the position index.

    """

    # Index in the pressure level of query matrix
    coords_zi = torch.arange(window_size[0])

    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(start=window_size[0]-1, end=-1, step=-1) * window_size[0]
  
    # Index in the latitude of query matrix
    coords_hi = torch.arange(window_size[1])

    # Index in the latitude of key matrix
    coords_hj = -torch.arange(start=window_size[1]-1, end=-1, step=-1) * window_size[1]
  
    # Index in the longitude of the key-value pair
    coords_w = torch.arange(window_size[2])
  
    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w], indexing='ij'))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w], indexing='ij'))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1) 
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0)

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += window_size[2] - 1
    coords[:, :, 1] *= 2 * window_size[2] - 1
    coords[:, :, 0] *= (2 * window_size[2] - 1) * (window_size[1]**2)

    # Sum up the indexes in three dimensions
    position_index = torch.sum(coords, dim=-1)

    # Flatten the position index to facilitate further indexing
    position_index = torch.flatten(position_index)
    
    return position_index

#@lru_cache
def create_mask_attention(window_size, displacement, n_t, n_h, n_w):
    img_mask = torch.zeros((1, n_t, n_h, n_w, 1))  # 1 P H W 1
    t_slices = (slice(0, -window_size[0]),
                slice(-window_size[0], -displacement[0]),
                slice(-displacement[0], None))
    h_slices = (slice(0, -window_size[1]),
                slice(-window_size[1], -displacement[1]),
                slice(-displacement[1], None))
    w_slices = (slice(0, -window_size[2]),
                slice(-window_size[2], -displacement[2]),
                slice(-displacement[2], None))
    cnt = 0
    for t, h, w in itt.product(t_slices, h_slices, w_slices):
        img_mask[:, t, h, w, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
