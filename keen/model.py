"""Keen-1: compact decoder-only Transformer, implemented with tinygrad."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from tinygrad import Tensor, nn
from keen.tokenizer import VOCAB_SIZE

@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    context: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2
    def __post_init__(self):
        if self.vocab_size != VOCAB_SIZE or self.width % self.heads or min(self.context,self.width,self.heads,self.layers) < 1:
            raise ValueError('Invalid model configuration')

class Block:
    def __init__(self, c):
        self.heads = c.heads
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.qkv, self.proj = nn.Linear(c.width, c.width * 3), nn.Linear(c.width, c.width)
        self.fc1, self.fc2 = nn.Linear(c.width, c.width * 4), nn.Linear(c.width * 4, c.width)
    def __call__(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(y): return y.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        a = heads(q).scaled_dot_product_attention(heads(k), heads(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(self.fc1(self.norm2(x)).gelu())

class KeenModel:
    def __init__(self, config):
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm, self.output = nn.LayerNorm(config.width), nn.Linear(config.width, config.vocab_size, bias=False)
    def __call__(self, ids):
        b, t = ids.shape
        if t > self.config.context: raise ValueError('Context exceeded')
        x = self.token(ids) + self.position(Tensor.arange(t).reshape(1, t))
        for block in self.blocks: x = block(x)
        return self.output(self.norm(x))
    def embeddings(self, ids):
        # Mean pooled, learned token embeddings. Not a semantic retrieval model.
        return self.token(ids).mean(axis=1)

def save_model(model, directory):
    p = Path(directory); p.mkdir(parents=True, exist_ok=True)
    (p/'config.json').write_text(json.dumps(asdict(model.config), indent=2), encoding='utf-8')
    nn.state.safe_save(nn.state.get_state_dict(model), str(p/'model.safetensors'))

def load_model(directory):
    p = Path(directory)
    model = KeenModel(Config(**json.loads((p/'config.json').read_text(encoding='utf-8'))))
    nn.state.load_state_dict(model, nn.state.safe_load(str(p/'model.safetensors')), verbose=False)
    return model
