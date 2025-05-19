import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        """
        Initialize LayerNorm with given dimensions and bias option.
        PyTorch doesn't support simply bias=False
        """
        super().__init__()
        # gamma, beta
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # q, k, v
        self.c_in = nn.Linear(cfg.n_emb, cfg.n_emb*3, bias=cfg.bias)
        # output projection
        self.c_out = nn.Linear(cfg.n_emb, cfg.n_emb, bias=cfg.bias)

        # reg
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.flash = hasattr(torch.nn.functioanl, 'scaled_dot_product_attention')

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))
            
    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, )

        if self.flash:
            y = F.scale

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_in = nn.Linear(cfg.d_model, 3* cfg.d_model, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_out = nn.Linear(3* cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_in(x)
        x = self.gelu(x)
        x = self.c_out(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, dropout=cfg.dropout, bias=cfg.bias)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wtr = nn.Embedding(cfg.vocab_size, cfg.n_emb),
                wpe = nn.Embedding(cfg.block_size, cfg.n_emb),
                drop = nn.Dropout(cfg.dropout),
                h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f = nn.LayerNorm
            )
        )