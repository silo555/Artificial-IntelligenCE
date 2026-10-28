#!/bin/sh
set -eu
# Supply CEDEV=/path/to/CEdev when the toolchain is not already on PATH.
if [ -n "${CEDEV:-}" ]; then PATH="$CEDEV/bin:$PATH"; export PATH; fi
if ! command -v cedev-config >/dev/null 2>&1; then
    echo 'CE toolchain not found. Set CEDEV to its directory or add bin/ to PATH.' >&2
    exit 1
fi
cd "$(dirname "$0")/.."
make
