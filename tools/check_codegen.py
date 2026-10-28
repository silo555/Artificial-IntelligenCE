"""Check CE target assembly for general 32-bit helpers in the neural hot loop.

This is a structural check, not a cycle count or a hardware speed measurement.
"""
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('assembly', type=Path)
args = parser.parse_args()
text = args.assembly.read_text()
try: kernel = text.split('_rnn_step:')[1].split('.size\t_rnn_step')[0]
except IndexError: raise SystemExit('Cannot locate rnn_step in this assembly')
counts = {name: kernel.count('call\t' + name) for name in ('__lmulu','__ladd','__imulu')}
print(json.dumps({'kernel_assembly_lines':len(kernel.splitlines()), 'static_helper_call_sites':counts,
                  'calculator_speedup':None, 'note':'No runtime or cycle claim; benchmark on hardware.'},indent=2))
if counts['__lmulu'] or counts['__ladd']:
    raise SystemExit('Unexpected 32-bit arithmetic helper in inference hot loop')
