"""Regenerate every result table and figure from raw data: ``python -m beat.run_all``.

Order matters: B's T-only development selection feeds B's final models, which feed C.
"""
import time

from . import exp_a, exp_b, exp_c, figures


def main() -> None:
    start = time.perf_counter()
    exp_a.main([])
    exp_b.main(["dev"])
    exp_b.main(["final"])
    exp_c.main(["dev"])
    exp_c.main(["final"])
    # Sealed rejection rule, reported as a variant (post-seal deviation 7).
    exp_c.main(["dev", "--variant", "sealed_gate"])
    exp_c.main(["final", "--variant", "sealed_gate"])
    figures.main()
    print(f"All experiments regenerated in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
