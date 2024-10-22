import numpy as np

from .float_sequence import get_float_encoder, get_float_decoder
from .sympy_prefix import sympy_to_prefix, prefix_to_sympy

class Tokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.symbol2id = {}
        self.id2symbol = {}

        with open(vocab_path) as file:
            words = map(lambda x: x.rstrip('\n'), file.readlines())

        for (n, word) in enumerate(words):
            self.symbol2id[word] = n
            self.id2symbol[n] = word 

    def encode(self, lst):
        return np.array([[self.symbol2id[j] for j in i] for i in lst], dtype=np.ushort)

    def decode(self, lst):
        return [[self.id2symbol[j] for j in i] for i in lst]

class EncoderTokenizer(Tokenizer):
    def __init__(self, vocab_path, max_len=10):
        super().__init__(vocab_path)

        self.max_len = max_len

        self.float_encoder = get_float_encoder(2, 1, 100)
        self.float_decoder = get_float_decoder(1)

    def pre_tokenize(self, data):
        arr = np.array([i.split() for i in data], dtype=np.float32)
        permutation = [-1] + [i for i in range(arr.shape[1]-1)]
        arr = np.pad(arr[:, permutation], ((0,0), (0, self.max_len - arr.shape[1])), mode="constant", constant_values=[-np.inf])
        return arr
    
    def tokenize(self, data):
        out = self.pre_tokenize(data)
        out = self.float_encoder(out)
        out = self.encode(out)
        return out

class DecoderTokenizer(Tokenizer):
    def __init__(self, vocab_path):
        super().__init__(vocab_path)

    def equation_encoder(self, data):
        return [sympy_to_prefix(expr) for expr in data]
    
    def equation_decoder(self, data):
        return [prefix_to_sympy(lst) for lst in data]

    def pre_tokenize(self, data):
        return data
    
    def tokenize(self, data):
        out = self.pre_tokenize(data)
        out = self.equation_encoder(out)
        out = [['<bos>'] + i + ['<eos>'] for i in out]
        out = self.encode(out)
        return out
    
    def reverse_tokenize(self, data):
        out = self.decode(data)
        out = self.equation_decoder(out)
        return out