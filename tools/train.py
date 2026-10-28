"""Train a tiny hard-tanh character RNN locally with NumPy and truncated BPTT."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from model import NAMES, encode, appvar, IntegerRNN

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/demo.txt'))
    parser.add_argument('--output', type=Path, default=Path('models/demo'))
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--steps', type=int, default=18000)
    parser.add_argument('--sequence', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=.03)
    args = parser.parse_args()
    if not 8 <= args.hidden <= 128 or args.steps < 1 or not 4 <= args.sequence <= 256 or not 0 < args.learning_rate <= 1:
        parser.error('Invalid dimensions, steps, sequence length or learning rate')
    text = args.data.read_text(encoding='utf-8').lower().replace('\r\n', '\n')
    if any(c != '\n' and not 32 <= ord(c) <= 126 for c in text):
        parser.error('Training data must contain printable ASCII and newlines only')
    vocab = ''.join(sorted(set(text)))
    if not 2 <= len(vocab) <= 96 or len(text) < args.sequence + 2:
        parser.error('Corpus too short or alphabet outside 2..96 characters')
    rng = np.random.default_rng(args.seed)
    v, h = len(vocab), args.hidden
    ids = np.array([vocab.index(c) for c in text], dtype=np.int32)
    # Hold out the tail; split before selecting training windows.
    split = int(len(ids) * .9)
    if split <= args.sequence + 1:
        parser.error('Corpus too short for a 90/10 train/validation split')
    p = {'wx': rng.normal(0, .18, (h, v)), 'wh': rng.normal(0, .35 / np.sqrt(h), (h, h)),
         'bh': np.zeros(h), 'wy': rng.normal(0, .12, (v, h)), 'by': np.zeros(v)}
    memory = {k: np.zeros_like(value) for k, value in p.items()}
    start = time.monotonic(); losses = []
    for step in range(args.steps):
        offset = int(rng.integers(0, split - args.sequence - 1))
        x, y = ids[offset:offset + args.sequence], ids[offset + 1:offset + args.sequence + 1]
        states = [np.zeros(h)]; preacts = []; probs = []; loss = 0.
        for token, target in zip(x, y):
            a = p['wx'][:, token] + p['wh'] @ states[-1] + p['bh']
            state = np.clip(a, -1., 1.); states.append(state); preacts.append(a)
            logits = p['wy'] @ state + p['by']; logits -= logits.max()
            probability = np.exp(logits); probability /= probability.sum(); probs.append(probability)
            loss -= np.log(max(probability[target], 1e-20))
        gradients = {k: np.zeros_like(value) for k, value in p.items()}; dh = np.zeros(h)
        for t in reversed(range(args.sequence)):
            dy = probs[t].copy(); dy[y[t]] -= 1
            gradients['wy'] += np.outer(dy, states[t + 1]); gradients['by'] += dy
            da = (p['wy'].T @ dy + dh) * (np.abs(preacts[t]) < 1)
            gradients['wx'][:, x[t]] += da
            gradients['wh'] += np.outer(da, states[t]); gradients['bh'] += da
            dh = p['wh'].T @ da
        for key in NAMES:
            g = np.clip(gradients[key] / args.sequence, -1., 1.)
            memory[key] += g * g
            p[key] -= args.learning_rate * g / np.sqrt(memory[key] + 1e-8)
            np.clip(p[key], -1.95, 1.95, out=p[key])
        losses.append(loss / args.sequence)
        if step == 0 or (step + 1) % 250 == 0:
            print(f'{step + 1}/{args.steps}: training loss {np.mean(losses[-100:]):.4f}', flush=True)
    blob = encode(vocab, p); quantized = IntegerRNN(blob)
    # Evaluate exact exported integer logits on held-out next-character predictions.
    correct = total = 0; held_loss = 0.
    for token, target in zip(ids[split:-1], ids[split + 1:]):
        prediction = quantized.step(int(token)); correct += prediction == target; total += 1
        logits = quantized.logits.astype(float) / (64 * 127)
        logits -= logits.max(); held_loss += float(np.log(np.exp(logits).sum()) - logits[target])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.bin').write_bytes(blob)
    args.output.with_suffix('.8xv').write_bytes(appvar(blob))
    np.savez(args.output.with_suffix('.npz'), vocab=np.array(vocab), **p)
    report = {'architecture': 'character RNN, hard-tanh, Q6 int8 weights, Q7 state',
        'hidden': h, 'alphabet': v, 'model_bytes': len(blob), 'weight_parameters': sum(value.size for value in p.values()),
        'macs_per_step': h * h + h * v, 'steps': args.steps, 'seed': args.seed,
        'sequence': args.sequence, 'learning_rate': args.learning_rate, 'numpy_version': np.__version__,
        'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(), 'corpus_characters': len(text),
        'train_characters': split, 'validation_characters': total,
        'last_100_train_loss': float(np.mean(losses[-100:])),
        'quantized_validation_loss': held_loss / max(total, 1), 'quantized_validation_accuracy': correct / max(total, 1),
        'desktop_training_seconds': time.monotonic() - start,
        'warning': 'Tiny demonstration corpus. Validation is same-domain and may share phrases. Not evidence of general conversation or reasoning.',
        'calculator_speed': 'Not measured; desktop timing is not calculator timing'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    prompt = 'user: hello\nbot: '
    if all(c in vocab for c in prompt):
        print('Greedy integer sample:\n' + prompt + IntegerRNN(blob).generate(prompt, 160))
if __name__ == '__main__':
    main()
