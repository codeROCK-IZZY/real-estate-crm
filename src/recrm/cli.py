"""Command line: the call queue, reading an inquiry, contract deadlines, the console, and the evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recrm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue", help="who to call next in the demo brokerage")
    q.add_argument("--limit", type=int, default=10)
    r = sub.add_parser("read", help="read an inquiry (Claude with ANTHROPIC_API_KEY, rules otherwise)")
    r.add_argument("message")
    d = sub.add_parser("deadlines", help="a contract's deadlines")
    d.add_argument("contract", help="YYYY-MM-DD")
    d.add_argument("closing", help="YYYY-MM-DD")
    s = sub.add_parser("serve", help="the CRM console with the demo brokerage")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    ev = sub.add_parser("evaluate", help="call-queue policies on held-out and shifted brokerages")
    ev.add_argument("--out", default="eval/reports")
    ev.add_argument("--gate", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "evaluate":
        from .evaluate import run, save

        result = run()
        save(result, Path(args.out))
        print((Path(args.out) / "report.md").read_text(encoding="utf-8"))
        return 1 if args.gate and not result["ok"] else 0
    if args.cmd == "serve":
        import uvicorn

        from .app import demo_app

        uvicorn.run(demo_app(), host=args.host, port=args.port)
        return 0
    if args.cmd == "deadlines":
        from .deadlines import schedule

        try:
            tasks = schedule(dt.date.fromisoformat(args.contract), dt.date.fromisoformat(args.closing))
        except ValueError as err:
            print(err, file=sys.stderr)
            return 2
        for t in tasks:
            print(f"{t.due:%a %Y-%m-%d}  {t.label}")
        return 0
    if args.cmd == "read":
        import os

        from .app import AREAS
        from .intake import Reader, rules

        inquiry = (
            Reader().read(args.message, AREAS) if os.environ.get("ANTHROPIC_API_KEY") else rules(args.message, AREAS)
        )
        print(json.dumps(inquiry.__dict__, indent=2))
        return 0
    from .app import NOW_H, demo_console, name

    c = demo_console()
    for lead in c.queue()[: args.limit]:
        print(
            f"{name(lead):18s} {lead.source:12s} waiting {NOW_H - lead.arrive:5.1f} h  serious "
            f"{c.learned.serious(lead):.0%}  reachable {c.reachable(lead):.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
