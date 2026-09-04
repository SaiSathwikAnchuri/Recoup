"""Cross-platform task runner (there is no `make` on Windows).

    python tasks.py data          # regenerate data/ from config/priors.yaml
    python tasks.py train         # train + calibrate the cause classifier and the liquidity model
    python tasks.py harness       # run policies over the batch, print comparison
    python tasks.py audit         # write audit/audit_42.jsonl (per-case decision + reason + outcome)
    python tasks.py phase9        # oracle + ablations + prior sensitivity + fairness slice
    python tasks.py serve         # the closed-loop service + app: results overview + live console (http://127.0.0.1:8000)
    python tasks.py test          # run the test suite
    python tasks.py reproduce     # data + train + harness + audit + phase9 + tests
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
        rc = run(*PY, "-m", "agent.train_classifier")
        return rc or run(*PY, "-m", "agent.train_liquidity")
    if cmd == "harness":
        return run(*PY, "-m", "harness.run", "--seed", "42", *rest)
    if cmd == "audit":
        return run(*PY, "-m", "agent.audit", "--seed", "42", "--sqlite", *rest)
    if cmd == "phase9":
        return run(*PY, "-m", "experiments.phase9", *rest)
    if cmd == "serve":
        return run(*PY, "-m", "service.run", *rest)
    if cmd == "test":
        return run(*PY, "-m", "pytest", "-q")
    if cmd == "show":
        return run(*PY, "-m", "simulator.show", *rest)
    if cmd == "reproduce":
        rc = run(*PY, "-m", "simulator.generate", "--n", "400", "--seed", "42", "--out", "data")
        rc = rc or run(*PY, "-m", "agent.train_classifier")
        rc = rc or run(*PY, "-m", "agent.train_liquidity")
        rc = rc or run(*PY, "-m", "harness.run", "--seed", "42")
        rc = rc or run(*PY, "-m", "agent.audit", "--seed", "42", "--sqlite")
        rc = rc or run(*PY, "-m", "experiments.phase9")
        return rc or run(*PY, "-m", "pytest", "-q")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
