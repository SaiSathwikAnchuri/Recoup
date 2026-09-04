"""Launch the Recoup service.

    python -m service.run                 # http://127.0.0.1:8000
    python -m service.run --port 9000 --reload

Opens the operator console in a browser. Needs the models trained
(`python tasks.py train`) — the console tells you if they are missing.
"""

from __future__ import annotations

import argparse
import webbrowser

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not args.no_open:
        webbrowser.open(f"http://{args.host}:{args.port}/")
    uvicorn.run("service.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
