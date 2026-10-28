"""Evaluate lower-bit weight grids without assuming that packing makes the CE faster.

All runtime models retain the byte-weight format. --bits 4 means 15 symmetric
coefficient levels on a coarser Q6 grid, not a packed-nibble inference backend.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from model import NAMES, IntegerRNN, encode, appvar

def quantize(params, bits):
    if bits not in (4, 6, 8): raise ValueError('Choose 4, 6, or 8 bits')
    step = 1 << (8 - bits)
    maximum = (1 << (bits - 1)) - 1
    return {name: np.clip(np.rint(params[name] * 64 / step), -maximum, maximum) * step / 64 for name in NAMES}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/demo.txt'))
    parser.add_argument('--bits', type=int, choices=(4, 6, 8), default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    with np.load(args.checkpoint, allow_pickle=False) as file:
        vocab = str(file['vocab']); params = {k: file[k] for k in NAMES}
    text = args.data.read_text().lower().replace('\r\n','\n')
    text = text[int(.9*len(text)):]
    report = []
    for bits in (4, 6, 8) if args.compare else (args.bits,):
        blob = encode(vocab, quantize(params, bits)); model = IntegerRNN(blob)
        correct = total = 0; loss = 0.
        for a, b in zip(text, text[1:]):
            token, target = vocab.index(a), vocab.index(b)
            predicted = model.step(token); correct += predicted == target; total += 1
            logits = model.logits.astype(float) / (64*127); logits -= logits.max()
            loss += float(np.log(np.exp(logits).sum()) - logits[target])
        samples = {}
        for prompt in ('hello','who are you','what is a model','what is love'):
            query = 'user: ' + prompt + '\nbot: '
            if all(c in vocab for c in query): samples[prompt] = IntegerRNN(blob).generate(query, 120).split('\n')[0]
        report.append({'effective_coefficient_bits':bits, 'runtime_weight_bits':8, 'model_bytes':len(blob),
            'validation_loss':loss/max(total,1), 'validation_next_char_accuracy':correct/max(total,1),
            'samples':samples, 'calculator_speed':'unmeasured; no packed-nibble backend',
            'caveat':'Same repeated toy corpus. These are not generalization metrics.'})
        if args.output and bits == args.bits:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix('.bin').write_bytes(blob)
            args.output.with_suffix('.8xv').write_bytes(appvar(blob))
    print(json.dumps(report, indent=2))
if __name__ == '__main__': main()
