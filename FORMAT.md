# CERNN01 model format

All multibyte header integers are unsigned little-endian. The file is independent of compiler struct layout. Maximum accepted file length: 65,512 bytes.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `CERNN01` followed by a zero byte |
| 8 | 2 | hidden units H, 1..128 |
| 10 | 2 | alphabet size V, 2..96 |
| 12 | 4 | payload size, exactly V + 2HV + HH + H + V |
| 16 | 4 | FNV-1a 32-bit hash of payload, offset basis 2166136261, multiplier 16777619, wrapping modulo 2^32 |
| 20 | 4 | reserved flags, must be zero |
| 24 | V | unique ASCII alphabet bytes, newline or 32..126 |
| next | HV | input weights Wx, H rows by V columns |
| next | HH | recurrent weights Wh, H rows by H columns |
| next | H | recurrent bias bh |
| next | VH | output weights Wy, V rows by H columns |
| next | V | output bias by |

Weights/biases are row-major signed 8-bit two's-complement Q6 numbers. Export rounds float weights to the nearest representable value using NumPy's ties-to-even rounding. State values represent h/127 and are clamped to [-127,127]. Output scores are signed 32-bit values. The eZ80 engine uses native signed 24-bit accumulators (the maximum magnitude is bounded by 2,113,280); the desktop reference uses signed 32-bit accumulators. The engine uses no floating-point arithmetic.

For an input character index x:

```text
s[i] = 127 * (Wx[i,x] + bh[i]) + sum_j(Wh[i,j] * old_h[j])
new_h[i] = clamp(round_half_away_from_zero(s[i] / 64), -127, 127)
score[k] = 127 * by[k] + sum_j(Wy[k,j] * new_h[j])
next = first index with maximum score
```

All new state values are computed before replacing old state. Ties in output scores choose the earliest vocabulary entry. No softmax is needed for greedy generation. For evaluating loss against the float training scale, scores are divided by 64*127 before softmax.

A new query resets state to zeros and feeds every character of `user: PROMPT\nbot: `. The next predicted character is emitted and fed back to generate the following character. The UI stops at the first newline or 240 characters. Neural history exists only within that query.

The hash detects accidental corruption; it is not an authenticity check or cryptographic signature. The loader independently validates dimensions, alphabet, exact lengths, and flags before assigning weight pointers.

## TI container

The `.8xv` wraps the model as a single standard TI83F AppVar, archived by default, named `CEAIMDL`. Its variable data starts with the conventional two-byte data size followed by the model bytes. TI file framing includes matching variable lengths and the standard 16-bit additive checksum. `fileioc`'s `ti_GetDataPtr` returns the actual AppVar data after its OS length field, so the runtime sees `CERNN01` at offset zero.
