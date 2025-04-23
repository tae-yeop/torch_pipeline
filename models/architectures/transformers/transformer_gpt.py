import torch
import torch.nn as nn



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
            )
        )