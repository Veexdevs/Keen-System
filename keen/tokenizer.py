"""Reversible UTF-8 byte tokenizer; IDs 0-255 are bytes, 256/257/258 are special."""
import codecs

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259

def encode(text: str, bos: bool = False, eos: bool = False) -> list[int]:
    return ([BOS] if bos else []) + list(text.encode('utf-8')) + ([EOS] if eos else [])

def decode(tokens: list[int]) -> str:
    return bytes(i for i in tokens if 0 <= i < 256).decode('utf-8', errors='replace')

def incremental_decoder():
    return codecs.getincrementaldecoder('utf-8')(errors='replace')
