import numpy as np

import torch
import gc
gc.collect()
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_


from einops import rearrange

import itertools as itt
from functools import lru_cache

import torch
from einops import rearrange



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    windows = rearrange(x, 'b (nw_h w_h) (nw_w w_w) c -> (b nw_h nw_w) (w_h w_w) c', 
                        w_h=window_size[0], w_w=window_size[1])
    return windows


def window_reverse(windows, window_size, B, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = rearrange(windows, '(b nw_h nw_w) w_h w_w c -> b (nw_h w_h) (nw_w w_w) c',
                  nw_w = H // window_size[0], nw_h = W // window_size[1])
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
def create_mask_attention(window_size, displacement, n_h, n_w):
    img_mask = torch.zeros((1, n_h, n_w, 1))  # 1 P H W 1

    h_slices = (slice(0, -window_size[0]),
                slice(-window_size[0], -displacement[0]),
                slice(-displacement[0], None))
    w_slices = (slice(0, -window_size[1]),
                slice(-window_size[1], -displacement[1]),
                slice(-displacement[1], None))
    cnt = 0
    for h, w in itt.product(h_slices, w_slices):
        img_mask[:, h, w, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask



class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super(CyclicShift, self).__init__()
        self.displacement = displacement
    def forward(self, x):
        """
        Shifts the input tensor along each dimension according to the displacement vector.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Shifted tensor.
        """
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1]), dims=(1, 2))
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, attn_drop_rate, proj_drop_rate):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(attn_drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop_rate)
        )
    def forward(self, x, mask=None):
        return self.mlp(x)  
 
    
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, shifted, window_size, attn_drop_rate, proj_drop_rate, relative_pos_embedding, device):
        super(WindowAttention, self).__init__()
        # dim = C = 192
        # num_heads = 3
        # head_dim = 64
        self.dim = dim 
        inner_dim = head_dim * num_heads # for qkv
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.device = device


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        self.relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        
        
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True) 
        self.attn_drop = nn.Dropout(attn_drop_rate)   
        self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim),
                        nn.Dropout(proj_drop_rate)
        )
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
    def forward(self, x, mask=None) :           
        
        b, n_h, _, _ = x.shape
        x = window_partition(x, self.window_size)
        B_, N, C = x.shape
        
        qkv = self.to_qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # q shape (1860, 3, 144, 32)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, 
        q = q * self.scale
        
        #attn = q @ k.transpose(-2, -1)
        attn = torch.einsum('b h W c, b h w c-> b h W w', q, k)
        
        
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        
        
        if self.shifted :
            mask = mask.to(self.device) 
            attn += mask.repeat(b, 1, 1).unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        attn = torch.einsum('b h W w, b h W c-> b h W c', attn, v).reshape(B_, N, C)
        return self.to_out(attn)
    
    
    
    
class Encoder(nn.Module):
    def __init__(self, parameters, shifted):
        super(Encoder, self).__init__()
        
        self.num_heads = parameters['num_heads']
        self.dim = parameters['hidden_emb']
        self.head_dim = parameters['head_dim']
        self.shifted = shifted
        self.window_size = parameters['window_size']
        self.attn_drop_rate = parameters['attn_drop_rate']
        self.proj_drop_rate = parameters['proj_drop_rate']
        self.relative_pos_embedding = parameters['relative_pos_embedding']
        self.mlp_dim = parameters['mlp_dim']
        self.device = parameters['device']
        
        if self.shifted :
            self.displacement = (self.window_size[0] // 2, self.window_size[1] // 2)
            self.cyclic_shift = CyclicShift((-self.displacement[0], -self.displacement[1]))
            self.cyclic_back_shift = CyclicShift((self.displacement[0], self.displacement[1]))
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn_block = WindowAttention(dim = self.dim, 
                                                num_heads = self.num_heads,
                                                head_dim = self.head_dim,
                                                shifted = self.shifted, 
                                                window_size = self.window_size, 
                                                attn_drop_rate=self.attn_drop_rate,
                                                proj_drop_rate=self.proj_drop_rate,
                                                relative_pos_embedding = self.relative_pos_embedding, 
                                                device = self.device)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp_block = FeedForward(dim = self.dim, hidden_dim = self.mlp_dim, attn_drop_rate = self.attn_drop_rate, proj_drop_rate = self.proj_drop_rate)


    def forward(self, x):
        
        b, n_h, n_w, c = x.shape
        pad_w = (self.window_size[1] - n_h % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[1] - n_h % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        skip = x
        _, Hp, Wp,_ = x.shape
        
        # cyclic Shift     
        if self.shifted:
            shifted_x =  self.cyclic_shift(x)
            attn_mask = create_mask_attention(self.window_size, self.displacement, Hp, Wp)
        else : 
            shifted_x = x
            attn_mask = None
        
                
        # * W_MSA / WS_MSA
        attn_windows = self.attn_block(shifted_x, mask = attn_mask)
        attn_windows = attn_windows.view(-1, *(self.window_size+(c,)))
        
        shifted_x = window_reverse(attn_windows, self.window_size, b, Hp, Wp)  # B D' H' W' C
        
        if self.shifted : 
            x = self.cyclic_back_shift(shifted_x)
        else : 
            x = shifted_x
        
        x = self.norm1(x) + skip
        skip = x
        x = skip + self.norm2(self.mlp_block(x, None))
        return x
 
    

class StageModule(nn.Module):
    def __init__(self, parameters):
        super(StageModule, self).__init__()
        self.num_layers = parameters['num_layers']
        
        assert self.num_layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers // 2):
            self.layers.append(nn.ModuleList([
                Encoder(parameters, shifted = False),
                Encoder(parameters, shifted = True),
            ]))


    def forward(self, x):
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x  



class Down_Sampling(nn.Module):
    """Down-Sampling operation based on the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"""
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super(Down_Sampling, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        # Extract sub-pixels from x
        x0 = x[:, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]     

        # Concatenate sub-pixels along the last dimension
        x = torch.cat([x0, x1, x2, x3], -1)

        # Normalize the concatenated sub-pixels and apply reduction
        x = self.reduction(self.norm(x))
        return x


class Up_Sampling(nn.Module):
    """Up-Sampling operation based on the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"""
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super(Up_Sampling, self).__init__()
        self.dim = dim 
        self.expansion = nn.Linear(2*dim, 4*dim)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.expansion(x)
        splits = torch.split(x, self.dim, -1)
        out = torch.cat((torch.cat((splits[0], splits[1]), dim=2), torch.cat((splits[2], splits[3]), dim=2)), dim=3)
        return self.norm(out)



class PatchEmbedding(nn.Module):
    def __init__(self, parameters):
        super(PatchEmbedding, self).__init__()
        self.data_size = parameters['data_size']
        self.hidden_emb = parameters['hidden_emb']
        self.num_channels = parameters['nb_channels']
        self.kernel_size = parameters['kernel_size']
        self.norm_layer = parameters['norm_layer'](self.hidden_emb)


        self.patch_embedding = nn.Conv2d(
            self.num_channels, self.hidden_emb, kernel_size=self.kernel_size, 
            stride=self.kernel_size
            )
    
    def forward(self, x):
        
        x = rearrange(x, 'b h w c -> b c h w')
        _, _, n_h, n_w = x.shape
        
        if n_w % self.kernel_size[0] != 0:
            x = F.pad(x, (0, self.kernel_size[0] - n_w % self.kernel_size[0]))
        if n_h % self.kernel_size[1]!= 0:
            x = F.pad(x, (0, 0, 0, self.kernel_size[1] - n_h % self.kernel_size[1]))
        
        x = self.patch_embedding(x)
        
        ### Normalization
        _, _, n_h, n_w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm_layer(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=n_h, w=n_w)
        return x


class PactchRecovery(nn.Module):
    def __init__(self, parameters):
        super(PactchRecovery, self).__init__()
        self.dim = parameters['hidden_emb']
        self.num_channels = parameters['nb_channels']
        self.kernel_size = parameters['kernel_size']

        
        self.conv= nn.ConvTranspose3d(in_channels= self.dim, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, x):
        x = rearrange(x, 'b p h w c -> b c p h w')
        x = self.conv(x).permute(0,2,3,4,1)
        return  x



class SwinTransformer(nn.Module):
    def __init__(self, parameters):
        super(SwinTransformer, self).__init__()

        self.data_size = parameters['data_size']
        
        self.enc_layers = parameters['enc_layers']
        
        self.patching = PatchEmbedding(parameters)

        self.encode1 = nn.ModuleList() #The first 2 Encoders
        self.encode2 = nn.ModuleList() #The last 6 Encoders
        self.encode3 = nn.ModuleList()
        
        
        self.down_sampling = nn.ModuleList()
        

        for _ in range(self.enc_layers):
            self.encode1.append(StageModule(parameters=parameters))

        self.down_sampling.append(Down_Sampling(dim=parameters['hidden_emb']))
    
        
                
        parameters['hidden_emb'] = 2*parameters['hidden_emb']
        parameters['head_dim'] = 2*parameters['head_dim']
        
        for _ in range(self.enc_layers):
            self.encode2.append(StageModule(parameters=parameters))
            
        
        self.down_sampling.append(Down_Sampling(dim=parameters['hidden_emb']))
    
        
        parameters['hidden_emb'] = 2*parameters['hidden_emb']
        parameters['head_dim'] = 2*parameters['head_dim']
        
        for _ in range(self.enc_layers):
            self.encode3.append(StageModule(parameters=parameters))
            
                
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(parameters['hidden_emb'], parameters['nb_classes'])
        
    

    def forward(self, x):
        
        x = self.patching(x) # x = (P, H, W, C=96)

        for encode in self.encode1:
            x = encode(x)
        
        #down-sampling
        x = self.down_sampling[0](x)   # x = (P, H/2, W/2, 2C)
        
        for encode in self.encode2:
            x = encode(x)
            
        
        #down-sampling
        x = self.down_sampling[1](x)   # x = (P, H/2, W/2, 2C)

        for encode in self.encode3:
            x = encode(x)
        
        x = rearrange(x, 'b w h c -> b (w h) c')
        
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
      

if __name__== '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(1, 56, 56, 3).to(device=device, dtype=torch.float32)
    
    
    parameters = {
        'kernel_size' : (4, 4),
        'hidden_emb' : 48, 
        'nb_classes' : 2, 
        'nb_channels' : 3, 
        'data_size' : (56, 56), 
        'norm_layer' : nn.LayerNorm,

        'num_layers' : 2,
        'num_heads' : 3,
        'head_dim' : 16, 
        'window_size' : (4, 4),
        'relative_pos_embedding'  : True,
        'attn_drop_rate' : 0.2,
        'proj_drop_rate' : 0.2, 
        'mlp_dim' : 48 * 2,
        'batch_size' : 2,
        'enc_layers' : 2,
        'device' : device
    }

    model = SwinTransformer(parameters).to(device=device, dtype=torch.float32)
    x = model(x)
    print('ddddd')
    