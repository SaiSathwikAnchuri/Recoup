# Recoup — the closed-loop service. Not required for the hackathon (`python
# tasks.py serve` is enough locally); this is here so "how would you deploy it"
# has a real answer instead of a hand-wave. No secrets baked in — every one of
# them (RAZORPAY_*, ANTHROPIC_API_KEY, RECOUP_API_KEY) is supplied at `docker run`
# time via -e / --env-file, exactly like running it bare.
FROM python:3.12-slim

WORKDIR /app

# system deps: none beyond what scikit-learn/numpy wheels already need on slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the service needs trained models to answer /decide, /webhook, /demo/*; bake
# them into the image at build time so a fresh container is ready immediately,
# the same way `python tasks.py train` is a one-time step locally.
RUN python -m simulator.generate --n 400 --seed 42 --out data \
 && python -m agent.train_classifier \
 && python -m agent.train_liquidity

# service/recoup.db lives on a volume in any real deployment — this default
# path inside the image is fine for a single throwaway container, not for
# anything you'd restart.
VOLUME ["/app/service"]

EXPOSE 8000
ENV RECOUP_EXECUTE_MODE=dry_run

CMD ["python", "-m", "service.run", "--host", "0.0.0.0", "--port", "8000", "--no-open"]
