# Implementação do Transformer Decoder-only:
# embedding
# positional encoding
# 4 a 6 camadas de decoder
# multi-head attention
# layer norm
# MLP position wise
# saída linear para o vocabulário

import torch
import torch.nn as nn
import math

# --------------------------------------------
# Positional Encoding (padrão transformer)
# --------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# --------------------------------------------
# Multi-head Self-Attention com máscara causal
# --------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # máscara causal (impede ver o futuro)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v   # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.W_o(out)


# --------------------------------------------
# Feed Forward MLP
# --------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------
# Um bloco de decoder
# --------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# --------------------------------------------
# Modelo final decoder-only
# --------------------------------------------

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, num_heads=4, max_len=4096):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads) 
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        x = self.pos(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
