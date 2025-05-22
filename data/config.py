from typing import TypedDict, Literal


class ModelConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate_emb: float
    drop_rate_attn: float
    drop_rate_shortcut: float
    qkv_bias: bool


GPT_CONFIG_124M: ModelConfig = {
    # Vocabulary size.
    'vocab_size': 50257,
    # Context length.
    'context_length': 1024,
    # Embedding dimension.
    'emb_dim': 768,
    # Number of attention heads.
    'n_heads': 12,
    # Number of layers.
    'n_layers': 12,
    # Dropout rates.
    'drop_rate_emb': 0.1,
    'drop_rate_attn': 0.1,
    'drop_rate_shortcut': 0.1,
    # Query-Key-Value bias.
    'qkv_bias': False
}


def get_config(
        model_name: Literal["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"] = "gpt2-small",
) -> ModelConfig:
    config = GPT_CONFIG_124M
    if model_name == "gpt2-small":
        config['emb_dim'] = 768
        config['n_layers'] = 12
        config['n_heads'] = 12
    elif model_name == "gpt2-medium":
        config['emb_dim'] = 1024
        config['n_layers'] = 24
        config['n_heads'] = 16
    elif model_name == "gpt2-large":
        config['emb_dim'] = 1280
        config['n_layers'] = 36
        config['n_heads'] = 20
    elif model_name == "gpt2-xl":
        config['emb_dim'] = 1600
        config['n_layers'] = 48
        config['n_heads'] = 25
    else:
        raise ValueError(f"Unknown model name `{model_name}`")

    return config
