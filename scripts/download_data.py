"""Download BNCI2014-001 (BCI Competition IV 2a) .mat files.

The BNCI Horizon website was unreachable on 2026-11-10, so files come from the
TU Graz host that MOABB also uses for the same dataset.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2014/"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "bnci2014_001"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for subject in range(1, 10):
        for session in "TE":
            name = f"A{subject:02d}{session}.mat"
            target = DATA_DIR / name
            if not target.exists():
                partial = target.with_suffix(".part")
                urllib.request.urlretrieve(BASE_URL + name, partial)
                partial.replace(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
            print(lines[-1], flush=True)
    (DATA_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
