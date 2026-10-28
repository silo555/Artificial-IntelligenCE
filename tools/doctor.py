"""Check local prerequisites without reading credentials or private files."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
path = os.environ.get('PATH', '')
if os.environ.get('CEDEV'): path = str(Path(os.environ['CEDEV'])/'bin') + os.pathsep + path
checks = {
    'python': sys.version.split()[0],
    'numpy_available': importlib.util.find_spec('numpy') is not None,
    'clang_available_for_desktop_tests': shutil.which('clang') is not None,
    'ce_toolchain_available': shutil.which('cedev-config', path=path) is not None,
    'trained_model_available': (ROOT/'models/demo.bin').exists(),
    'calculator_build_available': (ROOT/'bin/CEAI.8xp').exists(),
    'runtime_libraries_available': (ROOT/'outputs/dependencies/clibs.8xg').exists(),
    'physical_calculator_verified': False,
    'next_hardware_step': 'Launch CEAI and photograph the GRAPH benchmark screen.'}
print(json.dumps(checks, indent=2))
if not checks['numpy_available']: print('Install requirements.txt in a local virtual environment.')
if not checks['ce_toolchain_available']: print('Optional for rebuilding: install CE C Toolchain v15.0 and set CEDEV.')
if not checks['runtime_libraries_available']: print('For packaging: run python tools/fetch_libraries.py.')
