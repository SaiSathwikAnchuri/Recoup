# Convenience targets (Linux/macOS). On Windows use: python tasks.py <target>
PY ?= python

.PHONY: data train harness test reproduce show

data:
	$(PY) -m simulator.generate --n 400 --seed 42 --out data

train:
	$(PY) -m agent.train_classifier
	$(PY) -m agent.train_liquidity

harness:
	$(PY) -m harness.run --seed 42

test:
	$(PY) -m pytest -q

reproduce: data train harness test

show:
	$(PY) -m simulator.show $(CASE)
