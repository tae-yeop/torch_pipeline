import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.c_attn = nn.Linear(cfg.n_emb, cfg.n_emb * 3, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_emb, cfg.n_emb, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "mask", 
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
            )
        self.n_head = cfg.n_head
        self.n_emb = cfg.n_emb

    def forward(self, x):
        B, T, C = x.size()

        # [B, T, C] -> [B, T, 3C]
        q, k, v = self.c_attn(x).split(self.n_emb, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0
            )
        else:
            att = (q@k.tranpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_emb, cfg.mlp_expand * self.n_emb, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.mlp_expand * self.n_emb, self.n_emb, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.block_type == 'standard':
            self.ln_1 = nn.LayerNorm(cfg.n_emb)
            self.attn = nn.MultiheadAttention(
                embed_dim=cfg.n_emb, num_heads=cfg.n_head, batch_first=True
            )
            self.mlp = MLP(cfg)
            self.ln_2 = nn.LayerNorm(self.n_emb)
            
        elif cfg.block_type == 'manual':
            self.ln_1 = LayerNorm(cfg.n_emb, bias=cfg.bias)
            self.attn = CausalSelfAttention(cfg)
            self.ln_2 = LayerNorm(cfg.n_emb, bias=cfg.bias)
            self.mlp = MLP(cfg)


        self.block_type = cfg.block_type

    def forward(self, x):
        if self.block_type == 'standard':
            x_attn, _ = self.attn(x, x, x, is_causal=True, attn_mask=torch.empty(1,1), need_weights=False)
            x = self.ln_1(x_attn + x)
            x_mlp = self.mlp(x)
            x = self.ln_2(x_mlp + x)

        elif self.block_type == 'manual':
            # pre-normalization
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self, 
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wtr = nn.Embedding(cfg.vocab_size, cfg.n_emb), # embedding layer
                wpe = nn.Embedding(cfg.block_size, cfg.n_emb), # positional embedding
                drop = nn.Dropout(cfg.dropout), # dropout layer
                blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]), # transformer blocks
                ln_f = LayerNorm(cfg.n_emb, bias=cfg.bias), # final layer norm
            )
        )

        self.lm_head = nn.Linear(cfg.n_emb, cfg.vocab_size, bias=False) # output layer

        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, params in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(params, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx: (B, T)
        device = idx.device
        b, t, = idx.size()
        assert t <= self.cfg.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(idx) # shape (B, T, C)
        pos_emb = self.transformer.wpe(pos) # shape (T, C)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, -1, :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
        

