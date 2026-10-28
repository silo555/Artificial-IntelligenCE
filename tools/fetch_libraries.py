"""Fetch the official v15.0 runtime libraries and license, with pinned hashes."""
import hashlib
from pathlib import Path
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'clibs.8xg': ('https://github.com/CE-Programming/libraries/releases/download/v15.0/clibs.8xg',
                 '450bdefb41fe263f8d2d161b7cf431bbf99711f2ef78f02f796d530ed12dfbf0'),
    'CE-LIBRARIES-LICENSE.txt': ('https://raw.githubusercontent.com/CE-Programming/libraries/v15.0/license',
                 '8873672f0600ee0db35962f3d15b599f80bd9058d3290306c3d1bc9bcf954af2')}
if __name__ == '__main__':
    destination = ROOT / 'outputs/dependencies'; destination.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in FILES.items():
        target = destination / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
            print(name + ': verified existing file'); continue
        with urlopen(url, timeout=30) as response: data = response.read(2_000_000)
        if hashlib.sha256(data).hexdigest() != checksum:
            raise SystemExit('Download hash mismatch: ' + name)
        target.write_bytes(data)
        print(name + ': downloaded and verified')
