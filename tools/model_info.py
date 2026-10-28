"""Inspect model dimensions, storage and per-character operation count."""
import argparse
import json
from pathlib import Path
from model import decode
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
args = parser.parse_args()
blob = args.model.read_bytes(); vocab, p = decode(blob)
h, v = len(p['bh']), len(vocab)
print(json.dumps({'hidden': h, 'vocabulary': repr(vocab), 'alphabet_size': v,
    'model_storage_bytes': len(blob), 'weight_bytes': sum(a.size for a in p.values()),
    'macs_per_character': h*h + h*v,
    'recurrent_state_bytes_used': h, 'logit_bytes_used': 4*v,
    'engine_allocates': 'fixed 128-byte state + 384-byte logits + model pointers; 128-byte step scratch; compiler padding/stack additional',
    'weight_access': 'direct read-only pointer into the model AppVar; no full model RAM copy',
    'calculator_characters_per_second': None, 'status': 'requires physical benchmark'}, indent=2))
