"""The cache directory. Everything the code computes is stored under `artifacts/` (reproduce.sh links it to
CACHE_DIR). A step's output is read when present and computed otherwise; MANIFEST.sha256 lists every file the
report steps read, so a cache can be checked before use."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path('artifacts')
NARCBENCH = ROOT/'narcbench'                          # NARCBench's released dataset (scenarios, activations)
PRIMARY = ROOT/'maj/2026-09-22-repro'                 # per model: raw activations, rounds, labels, decompositions
STAGED = ROOT/'cache_expansion'                       # per model: raw activations with their row identities
ROUNDS = ROOT/'rounds'                                # per model: every Stego discussion round, replayed
OBSERVATIONS = ROOT/'observations'                    # J-PACT sweep outputs
PROBE = ROOT/'narc_baseline_b200'                     # NARCBench probe baseline outputs
INTERP = ROOT/'interp'                                # DeepSeek readouts and interventions
SCENARIOS = NARCBENCH/'scenarios'                     # transcripts; deepseek_v4_flash/ holds DeepSeek's own
MANIFEST = ROOT/'MANIFEST.sha256'


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify() -> bool:
    """True if MANIFEST.sha256 exists and every file it lists has its recorded hash."""
    if not MANIFEST.exists():
        return False
    for line in MANIFEST.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if not Path(path).is_file() or sha256(path) != digest:
            print(f'cache: {path} missing or changed')
            return False
    return True


def write_manifest(paths) -> None:
    MANIFEST.write_text(''.join(f'{sha256(p)}  {p}\n' for p in sorted(set(map(str, paths)))))


if __name__ == '__main__':
    # python cache.py              exit 0 if the cache verifies
    # python cache.py write LIST   write MANIFEST.sha256 for the paths listed (one per line) in LIST
    import sys
    if sys.argv[1:2] == ['write']:
        write_manifest(Path(sys.argv[2]).read_text().split())
    else:
        sys.exit(0 if verify() else 1)
