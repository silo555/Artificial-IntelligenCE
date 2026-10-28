"""Meaningful cross-language parity, malformed-model, and export tests."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from quantize import quantize
from model import IntegerRNN, NAMES, HEADER, encode, decode, appvar, shapes, fnv

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.binary = cls.folder / 'ceai'
        subprocess.run(['clang', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O1', '-g',
            '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(ROOT / 'src/rnn.c'),
            str(ROOT / 'tools/desktop.c'), '-o', str(cls.binary)], check=True)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def random_model(self, h=8, v=12, seed=3):
        vocab = '\n' + ''.join(chr(32 + i) for i in range(v-1))
        rng = np.random.default_rng(seed)
        params = {name: rng.integers(-127, 128, size=shape).astype(float) / 64
                  for name, shape in zip(NAMES, shapes(h, v))}
        return encode(vocab, params)
    def run_c(self, blob, prompt, count=0, trace=False):
        path = self.folder / 'model.bin'; path.write_bytes(blob)
        result = subprocess.run([str(self.binary), str(path), prompt, str(count)] + (['--trace'] if trace else []),
                                text=True, capture_output=True)
        self.assertNotIn('runtime error:', result.stderr)
        self.assertNotIn('AddressSanitizer', result.stderr)
        return result
    def test_exact_state_and_logit_parity_including_max_dimensions(self):
        for h, v, seed in [(1, 2, 9), (8, 12, 3), (64, 33, 4), (96, 64, 8), (128, 96, 11)]:
            blob = self.random_model(h, v, seed); model = IntegerRNN(blob)
            prompt = model.vocab[:min(v, 8)]
            result = self.run_c(blob, prompt, 15, True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = []; token = 0
            def feed(token):
                next_token = model.step(token)
                expected.append([token, model.h.tolist(), model.logits.tolist()])
                return next_token
            for char in prompt: token = feed(model.vocab.index(char))
            for _ in range(15): token = feed(token)
            self.assertEqual([json.loads(line) for line in result.stdout.splitlines()], expected)
    def test_demo_generation_matches_trained_export(self):
        blob = (ROOT / 'models/demo.bin').read_bytes()
        prompt = 'user: hello\nbot: '
        result = self.run_c(blob, prompt, 120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, IntegerRNN(blob).generate(prompt, 120) + '\n')
    def test_rejects_corrupt_and_truncated_models(self):
        blob = self.random_model()
        cases = [blob[:n] for n in [0, 4, 23, 24, len(blob)-1]] + [blob + b'junk']
        bad = bytearray(blob); bad[-1] ^= 1; cases.append(bytes(bad))
        for offset, data in [(0, b'BADMAGIC'), (8, struct.pack('<H', 129)), (10, struct.pack('<H', 97)), (20, b'\1\0\0\0')]:
            bad = bytearray(blob); bad[offset:offset+len(data)] = data; cases.append(bytes(bad))
        bad = bytearray(blob); bad[25] = bad[24]; bad[16:20] = struct.pack('<I', fnv(bad[24:])); cases.append(bytes(bad))
        for bad in cases:
            with self.assertRaises((ValueError, UnicodeDecodeError)): decode(bad)
            self.assertNotEqual(self.run_c(bad, ' ').returncode, 0)
    def test_appvar_roundtrip_and_ti_checksum(self):
        blob = self.random_model(); file = appvar(blob)
        self.assertEqual(file[:8], b'**TI83F*')
        self.assertEqual(struct.unpack_from('<H', file, 53)[0], len(file)-57)
        self.assertEqual(struct.unpack_from('<H', file, len(file)-2)[0], sum(file[55:-2]) & 65535)
        self.assertEqual(file[60:68].rstrip(b'\0'), b'CEAIMDL')
        self.assertEqual(file[69], 128)
        self.assertEqual(struct.unpack_from('<H', file, 72)[0], len(blob))
        self.assertEqual(file[74:-2], blob)
    def test_trainer_smoke_and_checkpoint_export(self):
        output = self.folder / 'smoke'
        subprocess.run([sys.executable, str(ROOT/'tools/train.py'), '--data', str(ROOT/'data/demo.txt'),
            '--output', str(output), '--steps', '8', '--hidden', '8', '--sequence', '16'],
            check=True, capture_output=True, text=True)
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as checkpoint:
            blob = encode(str(checkpoint['vocab']), {k: checkpoint[k] for k in NAMES})
        self.assertEqual(blob, output.with_suffix('.bin').read_bytes())
        self.assertEqual(appvar(blob), output.with_suffix('.8xv').read_bytes())
    def test_eight_bit_quantization_is_identical_and_lower_bits_use_expected_grids(self):
        with np.load(ROOT/'models/demo.npz', allow_pickle=False) as file:
            vocab = str(file['vocab']); params = {k: file[k] for k in NAMES}
        self.assertEqual(encode(vocab, quantize(params, 8)), (ROOT/'models/demo.bin').read_bytes())
        for bits, step in [(4, 16), (6, 4)]:
            _, q = decode(encode(vocab, quantize(params, bits)))
            for value in q.values(): self.assertTrue(np.all(value % step == 0))
    def test_accumulator_extremes_preserve_parity(self):
        h, v = 128, 2; vocab = ' ab'[:v]
        for sign in [-1, 1]:
            params = {name: np.full(shape, sign*127/64) for name, shape in zip(NAMES, shapes(h, v))}
            blob = encode(vocab, params); model = IntegerRNN(blob)
            result = self.run_c(blob, ' ', 5, True)
            self.assertEqual(result.returncode, 0, result.stderr)
            token = 0
            for trace in [json.loads(line) for line in result.stdout.splitlines()]:
                self.assertEqual(trace[0], token)
                token = model.step(token)
                self.assertEqual(trace[1], model.h.tolist()); self.assertEqual(trace[2], model.logits.tolist())
    def test_cli_rejects_unknown_prompt_character(self):
        result = self.run_c(self.random_model(), 'z')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Character not in model alphabet', result.stderr)
if __name__ == '__main__': unittest.main(verbosity=2)
