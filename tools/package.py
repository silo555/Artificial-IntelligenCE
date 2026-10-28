"""Make a calculator starter ZIP and a clean source/training toolkit ZIP."""
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs'

def check_ti(data):
    if len(data) < 57 or data[:8] != b'**TI83F*':
        raise ValueError('Not a standard TI variable container')
    if struct.unpack_from('<H', data, 53)[0] + 57 != len(data):
        raise ValueError('TI file length mismatch')
    if struct.unpack_from('<H', data, len(data)-2)[0] != sum(data[55:-2]) & 65535:
        raise ValueError('TI checksum mismatch')

def write_zip(filename, entries):
    manifest = {}
    with zipfile.ZipFile(OUTPUT / filename, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            data = path.read_bytes()
            if path.suffix.lower() in ('.8xp', '.8xv', '.8xg'): check_ti(data)
            info = zipfile.ZipInfo(name, (2026, 10, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix == '.sh' else 0o644) << 16
            archive.writestr(info, data)
            manifest[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(OUTPUT / filename) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP integrity failed')
    print(filename, (OUTPUT / filename).stat().st_size, 'bytes,', len(entries), 'files')
    return manifest

if __name__ == '__main__':
    OUTPUT.mkdir(exist_ok=True)
    starter = {'CEAI.8xp': ROOT/'bin/CEAI.8xp', 'demo.8xv': ROOT/'models/demo.8xv',
        'clibs.8xg': OUTPUT/'dependencies/clibs.8xg',
        'CE-LIBRARIES-LICENSE.txt': OUTPUT/'dependencies/CE-LIBRARIES-LICENSE.txt',
        'INSTALL.txt': ROOT/'INSTALL.txt', 'LICENSE': ROOT/'LICENSE'}
    toolkit = {name: ROOT/name for name in ['README.md', 'INSTALL.txt', 'FORMAT.md', 'LICENSE', 'requirements.txt', 'Makefile', '.gitignore']}
    for folder in ['src', 'tools', 'tests', 'data', 'models']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                toolkit[path.relative_to(ROOT).as_posix()] = path
    # No toolchains, venvs, ROMs, agent instructions, credentials, or personal exports.
    for name, path in toolkit.items():
        if path.suffix in ('.py', '.c', '.h', '.md', '.txt', '.json', '.sh'):
            text = path.read_text()
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,}|' + '/' + 'Users' + '/' + '|' + '-----BEGIN ' + r'.*PRIVATE KEY' + '-----', text):
                raise ValueError('Unexpected private content in ' + name)
    write_zip('CE-AI-starter.zip', starter)
    write_zip('CE-AI-toolkit.zip', toolkit)
