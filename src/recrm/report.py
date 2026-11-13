"""Markdown and HTML versions of the evaluation."""

from __future__ import annotations

import html
from typing import Any

BANK = {"tuning": "Tuning brokerages", "heldout": "Held-out brokerages", "shifted": "A shifted market (held back)"}
ARMS = ("fifo", "newest", "rules", "score", "score_retrained")


def markdown(r: dict[str, Any]) -> str:
    out = [
        "# Real estate CRM: evaluation",
        "",
        "Simulated brokerages over two months: more leads than calling time, leads that cool at different speeds by "
        "source, and a hidden seriousness the CRM has to infer from what it can see.",
        "",
    ]
    for key, label in BANK.items():
        b = r["banks"][key]
        out += [
            f"## {label}: {b['brokerages']} brokerages, {b['agents']} agents, {b['leads']:,} leads",
            "",
            "| Who is called first | Deals | Viewings | Serious leads never called | Median wait for a serious lead |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            if arm in b:
                a = b[arm]
                out.append(
                    f"| {a['label']} | {a['deals']:,} | {a['viewings']:,} | {a['serious_never_called']:,} of "
                    f"{a['serious_leads']:,} | {a['serious_median_wait_h']} h |"
                )
        out.append("")
    out += [
        "## Model",
        "",
        "Cooling time per source, learned from call history (hours for reachability to fall by e): "
        + ", ".join(f"{k} {v}" for k, v in r["cooling"].items()),
        "",
        "## Gate (regression limits, not targets)",
        "",
    ]
    for k, v in r["checks"].items():
        out.append(f"- {'PASS' if r['passed'][k] else 'FAIL'} {k}: {v} (limit {r['gate'][k]})")
    return "\n".join(out) + "\n"


CSS = """
:root{--bg:#0d1016;--panel:#151b24;--line:#263042;--ink:#e8edf4;--ink2:#97a3b6;--ok:#3ccf8e;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:34px 28px 56px}h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;
margin:28px 0 12px}.sub{color:var(--ink2);margin:0 0 20px}.tiles{display:grid;grid-template-columns:repeat(4,1fr);
gap:12px}.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}.tile b{
display:block;font-size:26px}.tile span{color:var(--ink2);font-size:13px}table{width:100%;border-collapse:collapse;
background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13.5px}th,td{
padding:8px 12px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:500;font-size:12px}.good{color:var(--ok);font-weight:600}.bad{color:var(--bad);
font-weight:600}@media(max-width:800px){.tiles{grid-template-columns:1fr 1fr}}
"""


def html_report(r: dict[str, Any]) -> str:
    e = html.escape
    h = r["banks"]["heldout"]
    s = r["banks"]["shifted"]
    tiles = [
        (
            f"+{100 * (h['score']['deals'] / max(1, h['newest']['deals']) - 1):.0f}%",
            f"more deals than calling the newest lead first ({h['score']['deals']:,} vs {h['newest']['deals']:,}, held-out)",
        ),
        (
            f"{h['score']['serious_never_called']:,} vs {h['newest']['serious_never_called']:,}",
            "serious leads never called: this CRM vs newest first",
        ),
        (f"{h['fifo']['deals']:,}", "deals when working the inbox oldest first: by then most leads have gone cold"),
        (
            f"{100 * (s['score_retrained']['deals'] / max(1, s['newest']['deals']) - 1):+.0f}%",
            f"vs newest first in a shifted market, retrained on the brokerage's own history (not retrained: "
            f"{100 * (s['score']['deals'] / max(1, s['newest']['deals']) - 1):+.0f}%)",
        ),
    ]
    out = [
        f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>Real estate CRM evaluation</title><style>{CSS}</style><main><h1>Real estate CRM: "
        "evaluation</h1><p class='sub'>Two months of simulated brokerages with more leads than calling time. The model "
        "and the cooling estimates were trained on separate tuning brokerages.</p><div class='tiles'>"
    ]
    out += [f"<div class='tile'><b>{e(v)}</b><span>{e(t)}</span></div>" for v, t in tiles]
    out.append("</div>")
    for key, label in BANK.items():
        if key == "tuning":
            continue
        b = r["banks"][key]
        best = max(b[a]["deals"] for a in ARMS if a in b)
        out.append(
            f"<h2>{e(label)}: {b['brokerages']} brokerages, {b['leads']:,} leads</h2><table><tr><th>Who is called "
            "first</th><th>Deals</th><th>Viewings</th><th>Serious leads never called</th><th>Median wait</th></tr>"
        )
        for arm in ARMS:
            if arm in b:
                a = b[arm]
                out.append(
                    f"<tr><td>{e(a['label'])}</td><td class='{'good' if a['deals'] == best else ''}'>"
                    f"{a['deals']:,}</td><td>{a['viewings']:,}</td><td>{a['serious_never_called']:,} of "
                    f"{a['serious_leads']:,}</td><td>{a['serious_median_wait_h']} h</td></tr>"
                )
        out.append("</table>")
    out.append(
        "<h2>Gate (regression limits)</h2><table><tr><th>Check</th><th>Value</th><th>Limit</th><th>Result</th></tr>"
    )
    for k, v in r["checks"].items():
        ok = r["passed"][k]
        out.append(
            f"<tr><td>{e(k)}</td><td>{v}</td><td>{r['gate'][k]}</td><td class='{'good' if ok else 'bad'}'>"
            f"{'pass' if ok else 'fail'}</td></tr>"
        )
    out.append("</table></main></html>")
    return "".join(out) + "\n"
