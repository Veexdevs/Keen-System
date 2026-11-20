"""Local autoregressive sampling. No provider API is called."""
import threading
import numpy as np
from tinygrad import Tensor
from keen.tokenizer import BOS, EOS, PAD, encode, incremental_decoder

class GenerationState:
    def __init__(self):
        self.finish_reason = 'length'

class Engine:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
    def tokens(self, prompt, max_tokens=64, temperature=0.8, top_k=40, state=None):
        ids = encode(prompt, bos=True)[-self.model.config.context:]
        rng = np.random.default_rng()
        with self.lock:
            for _ in range(max_tokens):
                x = Tensor(np.array([ids[-self.model.config.context:]], dtype=np.int32))
                logits = self.model(x).numpy()[0, -1].astype(np.float64)
                logits[BOS] = logits[PAD] = -np.inf
                if temperature == 0:
                    token = int(np.argmax(logits))
                else:
                    logits /= temperature
                    if top_k and top_k < len(logits):
                        keep = np.argpartition(logits, -top_k)[-top_k:]
                        masked = np.full_like(logits, -np.inf); masked[keep] = logits[keep]; logits = masked
                    probs = np.exp(logits - np.max(logits)); probs /= probs.sum()
                    token = int(rng.choice(len(probs), p=probs))
                if token == EOS:
                    if state is not None: state.finish_reason = 'stop'
                    break
                ids.append(token)
                yield token
    def text_chunks(self, *args, **kwargs):
        decoder = incremental_decoder()
        for token in self.tokens(*args, **kwargs):
            chunk = decoder.decode(bytes([token]), final=False)
            if chunk: yield chunk
        tail = decoder.decode(b'', final=True)
        if tail: yield tail
    def embeddings(self, texts):
        result = []
        with self.lock:
            for text in texts:
                ids = encode(text, bos=True)[-self.model.config.context:]
                result.append(self.model.embeddings(Tensor(np.array([ids], dtype=np.int32))).numpy()[0].tolist())
        return result
