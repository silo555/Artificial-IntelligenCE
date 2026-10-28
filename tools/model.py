"""Versioned model format and exact integer reference. No cloud services."""
import struct
from pathlib import Path
import numpy as np
MAGIC = b'CERNN01\0'
HEADER = struct.Struct('<8sHHIII')
NAMES = ('wx', 'wh', 'bh', 'wy', 'by')
def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value

def shapes(h, v):
    return ((h, v), (h, h), (h,), (v, h), (v,))

def encode(vocab, params):
    v, h = len(vocab), len(params['bh'])
    if not 1 <= h <= 128 or not 2 <= v <= 96:
        raise ValueError('Supported dimensions: hidden 1..128, alphabet 2..96')
    if len(set(vocab)) != v or any(c != '\n' and not 32 <= ord(c) <= 126 for c in vocab):
        raise ValueError('Alphabet must be unique printable ASCII or newline')
    payload = vocab.encode('ascii')
    for name, shape in zip(NAMES, shapes(h, v)):
        values = np.asarray(params[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Invalid tensor: ' + name)
        if np.max(np.abs(values)) > 127 / 64 + 1e-6:
            raise ValueError('Weights exceed Q6 range: ' + name)
        quantized = np.clip(np.rint(values * 64), -127, 127).astype('i1')
        payload += quantized.tobytes()
    data = HEADER.pack(MAGIC, h, v, len(payload), fnv(payload), 0) + payload
    if len(data) > 65512:
        raise ValueError('Model exceeds single-AppVar size limit')
    return data

def decode(data):
    if len(data) < HEADER.size:
        raise ValueError('Truncated model')
    magic, h, v, size, checksum, flags = HEADER.unpack_from(data)
    if magic != MAGIC or flags or not 1 <= h <= 128 or not 2 <= v <= 96:
        raise ValueError('Invalid model header')
    expected = v + sum(int(np.prod(shape)) for shape in shapes(h, v))
    if size != expected or len(data) != HEADER.size + size or len(data) > 65512 or fnv(data[HEADER.size:]) != checksum:
        raise ValueError('Invalid model size/checksum')
    vocab = data[HEADER.size:HEADER.size + v].decode('ascii')
    if len(set(vocab)) != v or any(c != '\n' and not 32 <= ord(c) <= 126 for c in vocab):
        raise ValueError('Invalid alphabet')
    offset, params = HEADER.size + v, {}
    for name, shape in zip(NAMES, shapes(h, v)):
        count = int(np.prod(shape))
        params[name] = np.frombuffer(data, dtype='i1', count=count, offset=offset).reshape(shape).astype(np.int32)
        offset += count
    return vocab, params

class IntegerRNN:
    def __init__(self, data):
        self.vocab, self.p = decode(data)
        self.h = np.zeros(len(self.p['bh']), dtype=np.int32)
        self.logits = None
    def step(self, token):
        if not 0 <= token < len(self.vocab):
            raise ValueError('Invalid token')
        p = self.p
        summed = 127 * (p['wx'][:, token] + p['bh']) + p['wh'] @ self.h
        rounded = np.where(summed >= 0, (summed + 32) // 64, -((-summed + 32) // 64))
        self.h = np.clip(rounded, -127, 127)
        self.logits = p['wy'] @ self.h + 127 * p['by']
        return int(np.argmax(self.logits))
    def generate(self, prompt, count=120):
        if not prompt:
            raise ValueError('Prompt must not be empty')
        token = 0
        for c in prompt:
            token = self.step(self.vocab.index(c))
        output = []
        for _ in range(count):
            output.append(self.vocab[token])
            token = self.step(token)
        return ''.join(output)

def appvar(data, name='CEAIMDL'):
    if not name.isascii() or not name.isalnum() or not name[0].isalpha() or not 1 <= len(name) <= 8:
        raise ValueError('AppVar names must use 1..8 ASCII letters/digits, starting with a letter')
    if len(data) > 65512:
        raise ValueError('AppVar payload too large')
    variable = struct.pack('<H', len(data)) + data
    entry = struct.pack('<HHB8sBBH', 13, len(variable), 21, name.encode().ljust(8, b'\0'), 0, 128, len(variable)) + variable
    header = b'**TI83F*\x1a\x0a\x00' + b'CE AI local neural model'.ljust(42, b'\0') + struct.pack('<H', len(entry))
    return header + entry + struct.pack('<H', sum(entry) & 65535)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path); parser.add_argument('--prompt', default='user: hello\nbot: ')
    parser.add_argument('--count', type=int, default=120)
    args = parser.parse_args()
    print(IntegerRNN(args.model.read_bytes()).generate(args.prompt.lower(), args.count))
