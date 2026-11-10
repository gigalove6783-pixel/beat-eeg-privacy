"""Regenerate every result table and figure from raw data: ``python -m beat.run_all``.

Order matters: B's T-only development selection feeds B's final models, which feed C.
"""
import time

from . import exp_a, exp_b, exp_c


def main() -> None:
    start = time.perf_counter()
    exp_a.main([])
    exp_b.main(["dev"])
    exp_b.main(["final"])
    exp_c.main(["dev"])
    exp_c.main(["final"])
    print(f"All experiments regenerated in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
