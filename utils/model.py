import torch
from torch import nn

from data.config import ModelConfig
from utils.attention import MultiHeadAttention


class GPTModel(nn.Module):

    def __init__(self, cfg: dict):
        super(GPTModel, self).__init__()

        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate_emb'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # Allow training on a CPU or GPU, depending on which device the input data sits on.
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        # Epsilon.
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate_attn'],
            qkv_bias=cfg['qkv_bias'],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate_shortcut'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut connection.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # Add the original input back.
        x = x + shortcut

        # Shortcut connection for feed forward block.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # Add the original input back.
        x = x + shortcut

        return x


def generate_text_simple(
        model: nn.Module,
        # Batch/array of indices in the current context.
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step: `(batch, n_token, vocab_size)` becomes `(batch, vocab_size)`.
        logits = logits[:, -1, :]
        # Probas has shape `(batch, vocab_size)`.
        probas = torch.softmax(logits, dim=-1)
        # Has shape `(batch, 1)`.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # Append sampled index to the running sequence, where `idx` has shape `(batch, n_tokens+1)`
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def get_device() -> torch.device:
    """Get device by priorities: cuda > mps > cpu"""
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        # "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
