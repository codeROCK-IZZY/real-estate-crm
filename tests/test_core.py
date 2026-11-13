"""The lead world, the model, the call queue and contract deadlines."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from recrm.deadlines import Calendar, schedule, us_holidays
from recrm.model import fit
from recrm.queue import POLICIES, Learned, estimate_cooling, simulate
from recrm.world import COOL_HOURS, FEATURE_NAMES, SOURCES, Brokerage, Lead, features


def small(seed: int) -> Brokerage:
    return Brokerage(seed, days=10, agents=3)


def test_world_is_deterministic_and_complete() -> None:
    a, b = small(3), small(3)
    assert [(x.source, x.message, x.intent) for x in a.leads] == [(x.source, x.message, x.intent) for x in b.leads]
    assert {x.source for x in a.leads} == set(SOURCES)
    assert all(x.arrive <= y.arrive for x, y in zip(a.leads, a.leads[1:], strict=False))
    assert len(features(a.leads[0])) == len(FEATURE_NAMES)


def test_fit_recovers_a_known_rule() -> None:
    g = np.random.default_rng(0)
    x = np.c_[np.ones(4000), g.normal(size=(4000, 2))]
    p = 1 / (1 + np.exp(-(x @ np.array([-0.5, 2.0, -1.0]))))
    y = (g.random(4000) < p).astype(float)
    w = fit(x, y, steps=3000).weights
    assert w[1] > 1.5 and -1.4 < w[2] < -0.6


def test_cooling_is_learned_from_calls_not_read_from_the_simulator() -> None:
    est = estimate_cooling([small(s) for s in (1, 2, 3)])
    assert set(est) == set(SOURCES)
    assert est["portal"] < est["website"] < est["referral"], est
    assert abs(est["portal"] - COOL_HOURS["portal"]) / COOL_HOURS["portal"] < 0.5


def test_every_policy_runs_with_the_same_capacity() -> None:
    b = small(4)
    learned = Learned(
        fit(np.array([features(x) for x in b.leads]), np.array([x.intent > 0.5 for x in b.leads], float)),
        {s: 48.0 for s in SOURCES},
    )
    calls = {p: simulate(b, p, learned)["calls"] for p in POLICIES}
    assert max(calls.values()) <= 3 * 3 * 4 * 10, calls  # 3 agents x 3 hours x 4 calls x 10 days


def test_oldest_first_starves_new_leads() -> None:
    b = small(5)
    assert simulate(b, "fifo")["serious_median_wait_h"] > simulate(b, "newest")["serious_median_wait_h"]


def test_priority_prefers_serious_and_still_warm_leads() -> None:
    model = fit(
        np.array([[1.0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]] * 50 + [[1.0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * 50),
        np.array([1.0] * 40 + [0.0] * 10 + [1.0] * 5 + [0.0] * 45),
    )
    ln = Learned(model, {s: 24.0 for s in SOURCES})
    hot = Lead(1, 0.0, "portal", "We're pre-approved and ready to move.", 1.0, 0, 0, True, 0.9)
    cold = Lead(2, 0.0, "portal", "Just browsing for now.", 1.0, 0, 0, False, 0.1)
    assert ln.priority(hot, 1.0) > ln.priority(cold, 1.0)
    assert ln.priority(hot, 1.0) > ln.priority(hot, 48.0), "the same lead is worth less once it has cooled"


def test_us_holidays_with_observed_days() -> None:
    h = us_holidays(2026)
    assert dt.date(2026, 7, 3) in h, "4 July 2026 is a Saturday, observed Friday"
    assert dt.date(2026, 11, 26) in h, "Thanksgiving"
    assert dt.date(2026, 5, 25) in h, "Memorial Day"
    assert dt.date(2027, 12, 31) in us_holidays(2027), "1 January 2028 is a Saturday, observed on 31 December"


def test_deadlines_count_business_days_and_roll_off_weekends() -> None:
    tasks = {t.key: t.due for t in schedule(dt.date(2026, 11, 20), dt.date(2026, 12, 26))}
    assert tasks["earnest"] == dt.date(2026, 11, 25), "Fri + 3 business days"
    assert tasks["inspection"] == dt.date(2026, 11, 30), "Sun 29th rolls to Monday"
    assert tasks["closing"] == dt.date(2026, 12, 28), "Saturday closing moves to Monday"
    assert tasks["walkthrough"] == dt.date(2026, 12, 24), "the business day before (Christmas Day is a holiday)"
    over_thanksgiving = {t.key: t.due for t in schedule(dt.date(2026, 11, 24), dt.date(2027, 1, 8))}
    assert over_thanksgiving["earnest"] == dt.date(2026, 11, 30), "Wed, Fri, Mon: Thursday 26th is Thanksgiving"


def test_deadlines_after_closing_are_refused() -> None:
    with pytest.raises(ValueError, match="after closing"):
        schedule(dt.date(2026, 11, 20), dt.date(2026, 12, 1))


def test_task_status() -> None:
    cal = Calendar.us(2026)
    t = schedule(dt.date(2026, 11, 2), dt.date(2026, 12, 15))[0]
    assert t.status(t.due - dt.timedelta(days=10), cal) == "upcoming"
    assert t.status(t.due, cal) == "due soon"
    assert t.status(t.due + dt.timedelta(days=1), cal) == "overdue"
    t.done = True
    assert t.status(t.due + dt.timedelta(days=1), cal) == "done"
