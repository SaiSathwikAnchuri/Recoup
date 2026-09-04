"""Recoup as a service — the body around the decision engine.

  ingestion   service/webhook.py   Razorpay `payment.failed` / `subscription.charged` -> a case
  decision    service/bridge.py    runs agent.ev_policy.EVPolicy on the case
  execution   service/executors.py retry / re-auth link / escalate  (dry-run, or Razorpay test mode)
  state       service/store.py     SQLite: cases, decisions, actions, escalations
  surface     service/app.py       FastAPI + an operator console at /

Nothing here changes the engine; it wraps it. Run: `python -m service.run` (or `python tasks.py serve`).
"""
