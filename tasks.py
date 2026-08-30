"""Cross-platform task runner (there is no `make` on Windows).

    python tasks.py data          # regenerate data/ from config/priors.yaml
    python tasks.py train         # train + calibrate the cause classifier
    python tasks.py harness       # run policies over the batch, print comparison
    python tasks.py test          # run the test suite
    python tasks.py reproduce     # data + train + harness + tests, the canonical end-to-end check
    python tasks.py show c0001    # dump one case
"""

import subprocess
import sys

PY = [sys.executable]


def run(*args: str) -> int:
    print("$", " ".join(args))
    return subprocess.call(list(args))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "reproduce"
    rest = sys.argv[2:]
    if cmd == "data":
        return run(*PY, "-m", "simulator.generate", "--n", "400", "--seed", "42", "--out", "data")
    if cmd == "train":
        return run(*PY, "-m", "agent.train_classifier", *rest)
    if cmd == "harness":
        return run(*PY, "-m", "harness.run", "--seed", "42", *rest)
    if cmd == "test":
        return run(*PY, "-m", "pytest", "-q")
    if cmd == "show":
        return run(*PY, "-m", "simulator.show", *rest)
    if cmd == "reproduce":
        rc = run(*PY, "-m", "simulator.generate", "--n", "400", "--seed", "42", "--out", "data")
        rc = rc or run(*PY, "-m", "agent.train_classifier")
        rc = rc or run(*PY, "-m", "harness.run", "--seed", "42")
        return rc or run(*PY, "-m", "pytest", "-q")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
