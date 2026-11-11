"""Regenerate the two small synthetic inputs: python examples/generate.py."""

from pathlib import Path

import pandas as pd

from beat_audit.synthetic import stream_arrays


def main():
    destination = Path(__file__).resolve().parent
    for kind in ("commands", "features"):
        data = stream_arrays(kind=kind)
        outputs = data.pop("outputs")
        if kind == "commands":
            data["outputs"] = outputs
        else:
            for i in range(outputs.shape[1]):
                data[f"outputs_{i}"] = outputs[:, i]
        pd.DataFrame(data).to_csv(destination / f"{kind}.csv", index=False)


if __name__ == "__main__":
    main()
