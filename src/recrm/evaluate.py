"""Evaluation: which leads agents call first, over two months of simulated brokerages.

    recrm evaluate [--gate] [--out DIR]

- `tuning` (brokerages 1-6): the model and the cooling estimates are trained on their history (a past period worked
  oldest-first), and the score's one setting was chosen here;
- `heldout` (brokerages 101-115): the same kind of market, run once after freezing, with the model as trained;
- `shifted` (brokerages 201-208): a market where portal leads are better, referrals worse, message words say less and
  browsing behaviour says more. Run with the model as trained, and with a model retrained on each brokerage's own
  earlier two months, which is what the CRM does when a brokerage has history of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import fit, history
from .queue import POLICIES, Learned, estimate_cooling, simulate
from .world import Brokerage, Shape

SHIFTED = Shape(source_effect=(0.2, 0.0, -0.3, -0.9, 0.2, 0.4), message_effect=0.5, engagement_effect=0.7)
LABELS = {
    "fifo": "Oldest first (inbox order)",
    "newest": "Newest first (speed to lead)",
    "rules": "Referrals, then urgent words, then newest",
    "score": "Predicted seriousness x reachability (this CRM)",
    "score_retrained": "The same, retrained on the brokerage's own history",
}


def train(brokers: list[Brokerage]) -> Learned:
    x, y = history(brokers)
    return Learned(fit(x, y), estimate_cooling(brokers))


def bank(brokers: list[Brokerage], learned: Learned, retrain_shape: Shape | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "brokerages": len(brokers),
        "agents": sum(b.agents for b in brokers),
        "leads": sum(len(b.leads) for b in brokers),
    }
    arms: list[tuple[str, str]] = [(p, p) for p in POLICIES]
    if retrain_shape is not None:
        arms.append(("score_retrained", "score"))
    for name, policy in arms:
        rows = []
        for b in brokers:
            ln: Learned | None = learned if policy == "score" else None
            if name == "score_retrained" and retrain_shape is not None:
                ln = train([Brokerage(b.seed + 5000, shape=retrain_shape, agents=b.agents)])
            rows.append(simulate(b, policy, ln))
        waits = [r["serious_median_wait_h"] for r in rows if r["serious_median_wait_h"] is not None]
        out[name] = {
            "label": LABELS[name],
            "deals": sum(r["deals"] for r in rows),
            "viewings": sum(r["viewings"] for r in rows),
            "serious_leads": sum(r["serious_leads"] for r in rows),
            "serious_never_called": sum(r["serious_never_called"] for r in rows),
            "serious_median_wait_h": round(float(np.median(waits)), 1) if waits else None,
            "calls": sum(r["calls"] for r in rows),
        }
    return out


GATE = {
    "heldout_deals_vs_newest_min": 1.10,
    "heldout_deals_vs_rules_min": 1.0,
    "shifted_deals_vs_newest_min": 0.95,
    "shifted_retrained_vs_newest_min": 1.0,
}


def run() -> dict[str, Any]:
    tuning = [Brokerage(s) for s in range(1, 7)]
    learned = train(tuning)
    banks = {
        "tuning": bank(tuning, learned),
        "heldout": bank([Brokerage(s) for s in range(101, 116)], learned),
        "shifted": bank([Brokerage(s, shape=SHIFTED) for s in range(201, 209)], learned, retrain_shape=SHIFTED),
    }
    h, s = banks["heldout"], banks["shifted"]
    checks = {
        "heldout_deals_vs_newest_min": round(h["score"]["deals"] / max(1, h["newest"]["deals"]), 3),
        "heldout_deals_vs_rules_min": round(h["score"]["deals"] / max(1, h["rules"]["deals"]), 3),
        "shifted_deals_vs_newest_min": round(s["score"]["deals"] / max(1, s["newest"]["deals"]), 3),
        "shifted_retrained_vs_newest_min": round(s["score_retrained"]["deals"] / max(1, s["newest"]["deals"]), 3),
    }
    passed = {k: v >= GATE[k] for k, v in checks.items()}
    return {
        "banks": banks,
        "model": json.loads(learned.model.to_json()),
        "cooling": learned.cooling,
        "checks": checks,
        "gate": GATE,
        "passed": passed,
        "ok": all(passed.values()),
    }


def save(result: dict[str, Any], out: Path) -> None:
    from .report import html_report, markdown

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "report.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    (out / "report.html").write_text(html_report(result), encoding="utf-8", newline="\n")
