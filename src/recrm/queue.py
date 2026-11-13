"""The call queue: who an agent rings next, and what that does over two months.

Policies (every one gets the same agents, hours and calls):
- `fifo`: oldest lead first, the order leads arrive in most CRMs' inbox;
- `newest`: newest first, the usual "speed to lead" advice;
- `rules`: referrals and past clients, then urgent wording, then newest;
- `score` (this CRM): predicted chance the lead is serious (the model) times how reachable they still are now, from
  how fast leads from that source have cooled in the brokerage's own call history.

Agents spend most of the day on viewings and paperwork: they have three calling hours a day (09:00, 13:00 and 17:00),
four calls an hour. A lead is dropped after four unanswered calls or two weeks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import Model
from .world import COOL_HOURS, SOURCES, Brokerage, Lead, features

POLICIES = ("fifo", "newest", "rules", "score")
CALL_HOURS = (9, 13, 17)
MAX_ATTEMPTS = 4
MAX_AGE_H = 14 * 24
CLOSE_RATE = 0.45


@dataclass
class Learned:
    model: Model
    cooling: dict[str, float]  # source -> hours for reachability to fall by e
    alpha: float = 0.0  # extra pull for faster-cooling sources (0 chosen on the tuning brokerages)
    _p: dict[int, float] = field(default_factory=dict)  # predictions for the brokerage being simulated, by lead id

    def serious(self, lead: Lead) -> float:
        key = lead.id
        if key not in self._p:
            self._p[key] = self.model.predict(features(lead))
        return self._p[key]

    def priority(self, lead: Lead, now: float) -> float:
        tau = self.cooling.get(lead.source, 48.0)
        return self.serious(lead) * float(np.exp(-(now - lead.arrive) / tau)) * tau ** -self.alpha


def reach_probability(lead: Lead, age: float) -> float:
    """The simulator's truth (never seen by the policies)."""
    return 0.8 * float(np.exp(-age / COOL_HOURS[lead.source])) * (1.0 if lead.phone else 0.6)


def _priority(policy: str, lead: Lead, now: float, learned: Learned | None) -> tuple[float, ...]:
    if policy == "fifo":
        return (-lead.arrive,)
    if policy == "newest":
        return (lead.arrive,)
    if policy == "rules":
        tier = 2 if lead.source in ("referral", "past_client") else 1 if "pre-approved" in lead.message or "move" in lead.message else 0
        return (tier, lead.arrive)
    assert learned is not None
    return (learned.priority(lead, now),)


def simulate(brokerage: Brokerage, policy: str, learned: Learned | None = None, seed: int = 0,
             log: list[tuple[str, float, bool, Lead]] | None = None) -> dict[str, Any]:
    """Run the two months; returns outcomes. With `log`, records every call (source, age, answered, lead)."""
    g = np.random.default_rng(seed)
    leads = copy.deepcopy(brokerage.leads)
    if learned is not None:
        learned._p.clear()  # lead ids are only unique within one brokerage
    arrived = 0
    open_leads: list[Lead] = []
    calls = 0
    for hour in range(brokerage.days * 24):
        while arrived < len(leads) and leads[arrived].arrive < hour + 1:
            open_leads.append(leads[arrived])
            arrived += 1
        if hour % 24 not in CALL_HOURS:
            continue
        for q in range(4):
            now = hour + q / 4
            open_leads = [x for x in open_leads if not (x.reached or x.lost) and now - x.arrive <= MAX_AGE_H]
            ready = [x for x in open_leads if x.arrive <= now]
            if not ready:
                continue
            ready.sort(key=lambda x: _priority(policy, x, now, learned), reverse=True)
            busy = 0
            for lead in ready[: brokerage.agents]:
                if busy >= brokerage.agents:
                    break
                busy += 1
                calls += 1
                age = now - lead.arrive
                if lead.first_contact is None:
                    lead.first_contact = now
                lead.attempts += 1
                answered = bool(g.random() < reach_probability(lead, age))
                if log is not None:
                    log.append((lead.source, age, answered, lead))
                if answered:
                    lead.reached = True
                    lead.viewing = bool(g.random() < lead.intent)
                    lead.closed = lead.viewing and bool(g.random() < CLOSE_RATE)
                elif lead.attempts >= MAX_ATTEMPTS:
                    lead.lost = True
    serious = [x for x in leads if x.intent >= 0.5]
    waits = [x.first_contact - x.arrive for x in serious if x.first_contact is not None]
    return {
        "leads": len(leads),
        "serious_leads": len(serious),
        "calls": calls,
        "reached": sum(x.reached for x in leads),
        "viewings": sum(x.viewing for x in leads),
        "deals": sum(x.closed for x in leads),
        "never_called": sum(x.first_contact is None for x in leads),
        "serious_never_called": sum(x.first_contact is None for x in serious),
        "serious_median_wait_h": round(float(np.median(waits)), 1) if waits else None,
        "serious_called_within_1h": round(sum(w <= 1 for w in waits) / max(1, len(serious)), 4),
    }


def simulate_history(brokerage: Brokerage, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """Training rows from a past period worked oldest-first: leads reached within two days, labelled by viewing."""
    log: list[tuple[str, float, bool, Lead]] = []
    simulate(brokerage, "fifo", seed=seed, log=log)
    rows = []
    for _src, age, answered, lead in log:
        if answered and age <= 48:
            rows.append((features(lead), 1.0 if lead.viewing else 0.0))
    return rows


def estimate_cooling(brokers: list[Brokerage], seed: int = 0) -> dict[str, float]:
    """Per source, the time constant that best explains which calls were answered at which lead age (grid MLE)."""
    calls: dict[str, list[tuple[float, bool, bool]]] = {s: [] for s in SOURCES}
    for b in brokers:
        log: list[tuple[str, float, bool, Lead]] = []
        simulate(b, "fifo", seed=seed, log=log)
        for src, age, answered, lead in log:
            calls[src].append((age, answered, lead.phone))
    grid = np.geomspace(4, 1000, 60)
    out = {}
    for src, rows in calls.items():
        if len(rows) < 30:
            out[src] = 48.0
            continue
        ages = np.array([r[0] for r in rows])
        ans = np.array([r[1] for r in rows], dtype=float)
        base = np.array([0.8 if r[2] else 0.48 for r in rows])  # phone vs email/text answer rates, measured once
        best, best_ll = 48.0, -np.inf
        for tau in grid:
            p = np.clip(base * np.exp(-ages / tau), 1e-6, 1 - 1e-6)
            ll = float(np.sum(ans * np.log(p) + (1 - ans) * np.log(1 - p)))
            if ll > best_ll:
                best, best_ll = float(tau), ll
        out[src] = round(best, 1)
    return out
