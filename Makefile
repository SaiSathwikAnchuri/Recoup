# Convenience targets (Linux/macOS). On Windows use: python tasks.py <target>
PY ?= python

.PHONY: data train harness audit phase9 serve test reproduce show

data:
	$(PY) -m simulator.generate --n 400 --seed 42 --out data

train:
	$(PY) -m agent.train_classifier
	$(PY) -m agent.train_liquidity

harness:
	$(PY) -m harness.run --seed 42

audit:
	$(PY) -m agent.audit --seed 42 --sqlite

phase9:
	$(PY) -m experiments.phase9

serve:
	$(PY) -m service.run

test:
	$(PY) -m pytest -q

reproduce: data train harness audit phase9 test

show:
	$(PY) -m simulator.show $(CASE)
