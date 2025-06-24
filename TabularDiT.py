import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class NumericalEmbedder(nn.Module):
    """
    Embeds Numerical Values.
    """
    def __init__(self, hidden_size, d_numerical):
        super().__init__()
        self.d_numerical = d_numerical
        self.embedding_table = nn.Embedding(d_numerical, hidden_size)
        self.scale_embedder = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, X_numerical):
        numerical_category = torch.arange(X_numerical.shape[1], device=X_numerical.device)
        numerical_category = numerical_category.unsqueeze(0).expand(X_numerical.shape[0], -1)[:, :X_numerical.shape[1]] # [batch, d_numerical]
        
        X_numerical = X_numerical.reshape(-1, self.d_numerical, 1)
        x_scale = self.scale_embedder(X_numerical)
        x_category = self.embedding_table(numerical_category)
        embeddings = x_scale + x_category
        
        return embeddings

class CategoricalEmbedder(nn.Module):
    """
    Embeds Numerical Values.
    """
    def __init__(self, hidden_size, categories):
        super().__init__()
        self.categories = categories
        self.embedding_table_list = nn.ModuleList([nn.Embedding(cat + 1, hidden_size) for cat in categories])
        
    def forward(self, X_categorical):
        num_category = len(self.categories)
        embedding_list = [self.embedding_table_list[i](X_categorical[:, [i]]) for i in range(num_category)]
        embeddings = torch.cat(embedding_list, dim=1)
        
        return embeddings



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, categories, d_numerical):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.numerical_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias = True),
                                              nn.SiLU(), 
                                              nn.Linear(hidden_size, hidden_size, bias = True),
                                              nn.SiLU(),
                                              nn.Linear(hidden_size, 1, bias = True),)
        self.categorical_linear_list = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size, bias = True), 
                                                                    nn.SiLU(),
                                                                    nn.Linear(hidden_size, hidden_size, bias = True),
                                                                    nn.SiLU(), 
                                                                    nn.Linear(hidden_size, cat + 1, bias = True))
                                                                    for cat in categories])
        self.categories = categories
        self.d_numerical = d_numerical

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        numerical_latent, categorical_latent = x[:, :self.d_numerical, :], x[:, self.d_numerical:, :]
        continuous_score = self.numerical_linear(numerical_latent).squeeze(-1)
        categorical_score_list = [self.categorical_linear_list[i](categorical_latent[:, i]) for i in range(len(self.categories))]
        return continuous_score, categorical_score_list


class TabularDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        categories,
        d_numerical,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        two_time=False
    ):
        super().__init__()
        self.two_time = two_time
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.two_time:
            self.s_embedder = TimestepEmbedder(hidden_size)
        
        self.categories, self.d_numerical = categories, d_numerical
        self.num_dim = d_numerical + len(categories)
        
        self.numerical_pos_embed = nn.Embedding(d_numerical, self.hidden_size)
        self.categorical_pos_embed = nn.Embedding(len(categories), self.hidden_size)
        nn.init.normal_(self.numerical_pos_embed.weight, std=0.02)
        nn.init.normal_(self.categorical_pos_embed.weight, std=0.02)
        
        self.numerical_embedder = NumericalEmbedder(hidden_size, d_numerical)
        self.categorical_embedder = CategoricalEmbedder(hidden_size, categories)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, categories, d_numerical)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # Initialize label embedding table:
        # Initialize timestep embedding MLP:
        
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.two_time:
            nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.s_embedder.mlp[2].weight, std=0.02)  
        
        nn.init.normal_(self.numerical_embedder.scale_embedder[0].weight, std=0.02)
        nn.init.normal_(self.numerical_embedder.scale_embedder[2].weight, std=0.02)
        nn.init.normal_(self.numerical_embedder.scale_embedder[4].weight, std=0.02)
    
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)


    def forward(self, X_numerical, X_categorical, t, s = None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        X_numerical_embedding = self.numerical_embedder(X_numerical)
        X_categorical_embedding = self.categorical_embedder(X_categorical)
        
        num_positions = torch.arange(X_numerical.shape[1], device=X_numerical.device)
        num_positions = num_positions.unsqueeze(0).expand(X_numerical.shape[0], -1)[:, :X_numerical.shape[1]]
        cat_positions = torch.arange(X_categorical.shape[1], device=X_categorical.device)
        cat_positions = cat_positions.unsqueeze(0).expand(X_categorical.shape[0], -1)[:, :X_categorical.shape[1]] # [batch, seq_len]
        num_positions = self.numerical_pos_embed(num_positions)
        cat_positions = self.categorical_pos_embed(cat_positions)
        
        X_numerical_embedding = X_numerical_embedding + num_positions
        X_categorical_embedding = X_categorical_embedding + cat_positions
        x = torch.cat([X_numerical_embedding, X_categorical_embedding], dim=1)

        t = self.t_embedder(t)                   # (N, D)
        if self.two_time and s is not None:
            s = self.s_embedder(s)               # (N, D)
            c = t + s
        else:
            c = t                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        final_output = self.final_layer(x, c)    # (N, T, patch_size ** 2 * out_channels)
        return final_output


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
