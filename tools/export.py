"""Export a locally trained NumPy checkpoint to a CE model and archived AppVar."""
import argparse
from pathlib import Path
import numpy as np
from model import NAMES, encode, appvar
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--name', default='CEAIMDL')
args = parser.parse_args()
with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
    blob = encode(str(checkpoint['vocab']), {k: checkpoint[k] for k in NAMES})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix('.bin').write_bytes(blob)
args.output.with_suffix('.8xv').write_bytes(appvar(blob, args.name))
print(f'Exported {len(blob)} model bytes; AppVar name {args.name}')
