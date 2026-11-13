"""The CRM console: who to call next and why, each lead's signals, the pipeline, transactions with their deadlines,
and reading new inquiries."""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .deadlines import Calendar, Task, schedule
from .intake import Inquiry, Reader, rules
from .model import fit, history
from .queue import Learned, estimate_cooling, simulate
from .world import SOURCES, Brokerage, Lead, features

e = html.escape
NOW_H = 20 * 24 + 12  # day 21, 12:00 in the demo brokerage
TODAY = dt.date(2026, 11, 13)
AREAS = ["Riverside", "Hillcrest", "Old Town", "Northgate", "Eastfield"]
FIRST = [
    "Maria",
    "James",
    "Aisha",
    "Wei",
    "Sofia",
    "Kwame",
    "Hannah",
    "Tomás",
    "Grace",
    "Oliver",
    "Leila",
    "Daniel",
    "Chloe",
    "Priya",
    "Noah",
    "Amara",
    "Lucas",
    "Mei",
    "Samuel",
    "Zara",
]
LAST = [
    "Lopez",
    "O'Brien",
    "Bello",
    "Chen",
    "Rossi",
    "Mensah",
    "Becker",
    "Silva",
    "Kim",
    "Grant",
    "Haddad",
    "Novak",
    "Martin",
    "Nair",
    "Reed",
    "Okafor",
    "Brook",
    "Lin",
    "Adeyemi",
    "Price",
]
SOURCE_LABEL = {
    "portal": "Property portal",
    "website": "Website",
    "open_house": "Open house",
    "social": "Social ad",
    "referral": "Referral",
    "past_client": "Past client",
}
STAGES = ("New", "Contacted", "Viewing booked", "Offer", "Under contract", "Closed")

CSS = """
:root{--bg:#0d1117;--panel:#151b24;--line:#263041;--ink:#e8edf4;--ink2:#97a3b6;--ok:#3ccf8e;--bad:#ff6b6b;--mid:#f5b54a;
--acc:#7aa2ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,
sans-serif}a{color:inherit}header{padding:14px 26px;border-bottom:1px solid var(--line);display:flex;gap:18px;
align-items:baseline;flex-wrap:wrap}header b{font-size:17px}header nav a{color:var(--ink2);text-decoration:none;
margin-right:14px}main{padding:20px 26px 48px;max-width:1260px;margin:0 auto}.grid{display:grid;
grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:14px 16px}.card b{display:block;font-size:24px}.muted{color:var(--ink2)}table{width:100%;
border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);
font-weight:500;font-size:12px}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.b{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-weight:600;white-space:nowrap}
.ok{background:#3ccf8e22;color:var(--ok)}.warn{background:#f5b54a22;color:var(--mid)}.bad{background:#ff6b6b22;
color:var(--bad)}.info{background:#7aa2ff22;color:var(--acc)}h1{font-size:22px;margin:0 0 4px}h2{font-size:13px;
color:var(--ink2);text-transform:uppercase;letter-spacing:.05em;margin:24px 0 8px}.bar{height:8px;border-radius:4px;
display:inline-block;vertical-align:middle}.pos{background:var(--ok)}.neg{background:var(--bad)}.board{display:grid;
grid-template-columns:repeat(6,minmax(160px,1fr));gap:10px;overflow-x:auto}.col{background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:10px}.col h3{margin:0 0 8px;font-size:13px}.chip{
border:1px solid var(--line);border-radius:8px;padding:7px 9px;margin-bottom:7px;font-size:12.5px}textarea,select,
input{background:#0f141c;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit}
textarea{width:100%;min-height:90px}button{background:var(--acc);color:#0b1020;border:0;border-radius:6px;
padding:8px 14px;font:inherit;font-weight:600;cursor:pointer}.msg{font-style:italic;color:#c9d3e0}
@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}}
"""


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>{e(title)}</title><style>{CSS}</style><header><b>Harbour &amp; Hill CRM</b><nav>"
        "<a href='/'>Call next</a><a href='/pipeline'>Pipeline</a><a href='/transactions'>Transactions</a>"
        f"<a href='/intake'>Read an inquiry</a></nav><span class='muted'>demo brokerage, day 21, 12:00</span></header>"
        f"<main>{body}</main></html>"
    )


def name(lead: Lead) -> str:
    return f"{FIRST[lead.id % len(FIRST)]} {LAST[(lead.id * 7 + lead.id // len(FIRST)) % len(LAST)]}"


@dataclass
class Deal:
    id: int
    lead: Lead
    address: str
    price: int
    contract: dt.date
    closing: dt.date
    tasks: list[Task] = field(default_factory=list)


@dataclass
class Console:
    brokerage: Brokerage
    learned: Learned
    leads: list[Lead]
    deals: list[Deal]
    reader: Reader | None = None

    def open_leads(self) -> list[Lead]:
        return [
            x for x in self.leads if x.arrive <= NOW_H and not x.reached and not x.lost and NOW_H - x.arrive <= 14 * 24
        ]

    def queue(self) -> list[Lead]:
        return sorted(self.open_leads(), key=lambda x: self.learned.priority(x, NOW_H), reverse=True)

    def stage(self, lead: Lead) -> str:
        if lead.closed:
            return STAGES[3 + lead.id % 3]
        if lead.viewing:
            return "Viewing booked"
        if lead.reached:
            return "Contacted"
        return "New"

    def reachable(self, lead: Lead) -> float:
        return float(np.exp(-(NOW_H - lead.arrive) / self.learned.cooling.get(lead.source, 48.0)))


def demo_console(reader: Reader | None = None) -> Console:
    tuning = [Brokerage(s) for s in range(1, 7)]
    x, y = history(tuning)
    learned = Learned(fit(x, y), estimate_cooling(tuning))
    b = Brokerage(42, agents=4)
    leads: list[Lead] = []
    simulate(b, "score", learned, hours=NOW_H, out=leads)
    g = np.random.default_rng(7)
    streets = ["Mill Lane", "Harbour Road", "Elm Grove", "Station Street", "Orchard Close", "Wharf Street"]
    deals = []
    for i, lead in enumerate([x for x in leads if x.closed][:6]):
        contract = TODAY - dt.timedelta(days=24 if i == 2 else int(g.integers(2, 26)))
        closing = contract + dt.timedelta(days=int(g.integers(30, 46)))
        d = Deal(
            i + 1,
            lead,
            f"{int(g.integers(2, 140))} {streets[i % 6]}, {AREAS[i % 5]}",
            int(g.integers(26, 90)) * 5000,
            contract,
            closing,
        )
        d.tasks = schedule(contract, closing)
        for t in d.tasks:
            t.done = t.due < TODAY and not (i == 2 and t.key == "appraisal")  # one deal has a late appraisal
        deals.append(d)
    return Console(b, learned, leads, deals, reader)


class LeadIn(BaseModel):
    message: str
    source: str = "website"
    views: int = 0
    saves: int = 0
    phone: bool = False
    budget_ratio: float = 1.0


class DealIn(BaseModel):
    contract: dt.date
    closing: dt.date


def score_new(c: Console, body: LeadIn) -> dict[str, Any]:
    if body.source not in SOURCES:
        raise HTTPException(422, f"source is one of {', '.join(SOURCES)}")
    inquiry = c.reader.read(body.message, AREAS) if c.reader is not None else rules(body.message, AREAS)
    lead = Lead(
        -1,
        NOW_H,
        body.source,
        body.message,
        max(body.budget_ratio, 0.01),
        max(0, body.views),
        max(0, body.saves),
        body.phone,
        0.0,
    )
    p = c.learned.model.predict(features(lead))
    return {
        "inquiry": inquiry.__dict__,
        "serious": round(p, 3),
        "why": [(k, round(v, 2)) for k, v in c.learned.model.explain(features(lead))[:4]],
    }


def create_app(c: Console) -> FastAPI:
    app = FastAPI(title="Real estate CRM")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "open_leads": len(c.open_leads()), "claude": c.reader is not None}

    @app.get("/api/queue")
    def api_queue(limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "lead": x.id,
                "name": name(x),
                "source": x.source,
                "waiting_h": round(NOW_H - x.arrive, 1),
                "serious": round(c.learned.serious(x), 3),
                "reachable": round(c.reachable(x), 3),
                "priority": round(c.learned.priority(x, NOW_H), 4),
            }
            for x in c.queue()[: max(1, min(limit, 200))]
        ]

    @app.post("/api/leads")
    def api_lead(body: LeadIn) -> dict[str, Any]:
        return score_new(c, body)

    @app.post("/api/transactions")
    def api_deal(body: DealIn) -> list[dict[str, str]]:
        try:
            return [{"task": t.label, "due": t.due.isoformat()} for t in schedule(body.contract, body.closing)]
        except ValueError as err:
            raise HTTPException(422, str(err)) from err

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        q = c.queue()
        today = [x for x in c.leads if x.first_contact is not None and NOW_H - 24 <= x.first_contact <= NOW_H]
        rows = []
        for x in q[:20]:
            why = c.learned.model.explain(features(x))[:2]
            reasons = "; ".join(f"{k} {'+' if v > 0 else '−'}" for k, v in why)
            rows.append(
                f"<tr><td><a href='/leads/{x.id}'>{e(name(x))}</a><div class='muted'>{e(SOURCE_LABEL[x.source])}"
                f"</div></td><td class='msg'>“{e(x.message)}”</td><td class='num'>{NOW_H - x.arrive:.0f} h</td>"
                f"<td class='num'>{c.learned.serious(x):.0%}</td><td class='num'>{c.reachable(x):.0%}</td>"
                f"<td class='muted'>{e(reasons)}</td></tr>"
            )
        body = (
            "<h1>Call next</h1><p class='muted'>Open leads ranked by how likely they are to be serious (learned from "
            "past leads) times how reachable they still are (how fast leads from that source go cold, learned from "
            "past calls). New inquiries from a portal cool within a day; referrals wait.</p><div class='grid'>"
            f"<div class='card'><b>{len(c.open_leads())}</b><span class='muted'>open leads not yet reached</span></div>"
            f"<div class='card'><b>{len(today)}</b><span class='muted'>first calls in the last 24 hours</span></div>"
            f"<div class='card'><b>{sum(x.viewing for x in c.leads)}</b><span class='muted'>viewings booked so far</span></div>"
            f"<div class='card'><b>{c.brokerage.agents}</b><span class='muted'>agents, three calling hours a day</span>"
            "</div></div><h2>Queue</h2><table><tr><th>Lead</th><th>Inquiry</th><th class='num'>Waiting</th>"
            "<th class='num'>Serious</th><th class='num'>Reachable now</th><th>Main signals</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        return _page("Call next", body)

    @app.get("/leads/{lead_id}", response_class=HTMLResponse)
    def lead_page(lead_id: int) -> str:
        if not 0 <= lead_id < len(c.leads) or c.leads[lead_id].arrive > NOW_H:
            raise HTTPException(404, "no such lead")
        x = c.leads[lead_id]
        parts = c.learned.model.explain(features(x))
        top = max((abs(v) for _, v in parts), default=1.0)
        bars = "".join(
            f"<tr><td>{e(k)}</td><td><span class='bar {'pos' if v > 0 else 'neg'}' style='width:"
            f"{int(120 * abs(v) / top)}px'></span> {v:+.2f}</td></tr>"
            for k, v in parts
        )
        ask = rules(x.message, AREAS)
        body = (
            f"<h1>{e(name(x))}</h1><p class='msg'>“{e(x.message)}”</p><div class='grid'><div class='card'><b>"
            f"{c.learned.serious(x):.0%}</b><span class='muted'>likely to be serious</span></div><div class='card'><b>"
            f"{c.reachable(x):.0%}</b><span class='muted'>still reachable ({e(SOURCE_LABEL[x.source])} leads cool in "
            f"about {c.learned.cooling.get(x.source, 48):.0f} h)</span></div><div class='card'><b>{e(c.stage(x))}"
            f"</b><span class='muted'>{x.attempts} call{'s' if x.attempts != 1 else ''} so far</span></div><div "
            f"class='card'><b>{x.views} / {x.saves}</b><span class='muted'>listings viewed / saved</span></div></div>"
            "<h2>Why the score</h2><table><tr><th>Signal</th><th>Effect on the odds</th></tr>" + bars + "</table>"
            f"<h2>Read from the inquiry</h2><p>Goal: <b>{e(ask.goal)}</b>"
            + (" · pre-approved" if ask.pre_approved else "")
            + (f" · moving within {ask.timeline_weeks} weeks" if ask.timeline_weeks else "")
            + "</p>"
        )
        return _page(name(x), body)

    @app.get("/pipeline", response_class=HTMLResponse)
    def pipeline() -> str:
        cols: dict[str, list[Lead]] = {s: [] for s in STAGES}
        for x in c.leads:
            if x.arrive <= NOW_H and not x.lost:
                cols[c.stage(x)].append(x)
        html_cols = []
        for stage, xs in cols.items():
            xs = sorted(xs, key=lambda x: -c.learned.serious(x))
            chips = "".join(
                f"<div class='chip'><a href='/leads/{x.id}'>{e(name(x))}</a><div class='muted'>"
                f"{e(SOURCE_LABEL[x.source])} · {c.learned.serious(x):.0%}</div></div>"
                for x in xs[:8]
            )
            more = f"<div class='muted'>and {len(xs) - 8} more</div>" if len(xs) > 8 else ""
            html_cols.append(
                f"<div class='col'><h3>{e(stage)} <span class='muted'>{len(xs)}</span></h3>{chips}{more}</div>"
            )
        return _page(
            "Pipeline",
            "<h1>Pipeline</h1><p class='muted'>Every lead by stage; within a stage, the most "
            "likely to be serious first.</p><div class='board'>" + "".join(html_cols) + "</div>",
        )

    @app.get("/transactions", response_class=HTMLResponse)
    def transactions() -> str:
        cal = Calendar.us(2026, 2027)
        rows = []
        for d in c.deals:
            nxt = next((t for t in d.tasks if not t.done), None)
            st = nxt.status(TODAY, cal) if nxt else "done"
            cls = {"overdue": "bad", "due soon": "warn", "upcoming": "info", "done": "ok"}[st]
            rows.append(
                f"<tr><td><a href='/transactions/{d.id}'>{e(d.address)}</a><div class='muted'>{e(name(d.lead))}"
                f"</div></td><td class='num'>${d.price:,}</td><td>{d.contract:%d %b}</td><td>{d.closing:%d %b}</td>"
                f"<td>{e(nxt.label) if nxt else 'All done'}{' · ' + nxt.due.strftime('%a %d %b') if nxt else ''}"
                f"</td><td><span class='b {cls}'>{e(st)}</span></td></tr>"
            )
        return _page(
            "Transactions",
            "<h1>Transactions</h1><p class='muted'>Deals under contract, with the next deadline "
            "from each contract's terms (business days skip weekends and US federal holidays).</p><table><tr>"
            "<th>Property</th><th class='num'>Price</th><th>Contract</th><th>Closing</th><th>Next</th>"
            "<th>Status</th></tr>" + "".join(rows) + "</table>",
        )

    @app.get("/transactions/{deal_id}", response_class=HTMLResponse)
    def transaction(deal_id: int) -> str:
        d = next((x for x in c.deals if x.id == deal_id), None)
        if d is None:
            raise HTTPException(404, "no such transaction")
        cal = Calendar.us(2026, 2027)
        rows = "".join(
            f"<tr><td>{t.due:%a %d %b %Y}</td><td>{e(t.label)}</td><td><span class='b "
            f"{ {'overdue': 'bad', 'due soon': 'warn', 'upcoming': 'info', 'done': 'ok'}[t.status(TODAY, cal)] }'>"
            f"{e(t.status(TODAY, cal))}</span></td></tr>"
            for t in d.tasks
        )
        return _page(
            d.address,
            f"<h1>{e(d.address)}</h1><p class='muted'>{e(name(d.lead))} · ${d.price:,} · contract "
            f"{d.contract:%d %b %Y}</p><h2>Deadlines</h2><table><tr><th>Due</th><th>Task</th><th>Status"
            "</th></tr>" + rows + "</table>",
        )

    def intake_page(message: str, result: dict[str, Any] | None, note: str) -> str:
        res = ""
        if result is not None:
            q = result["inquiry"]
            fields = [
                ("Goal", q["goal"]),
                ("Budget up to", f"${q['budget_max']:,}" if q["budget_max"] else "—"),
                ("Bedrooms", f"{q['bedrooms_min']}+" if q["bedrooms_min"] else "—"),
                ("Areas", ", ".join(q["areas"]) or "—"),
                ("Moving within", f"{q['timeline_weeks']} weeks" if q["timeline_weeks"] else "—"),
                ("Pre-approved", {True: "yes", False: "no", None: "—"}[q["pre_approved"]]),
            ]
            res = (
                "<h2>Read</h2><table>"
                + "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in fields)
                + f"<tr><th>Likely to be serious</th><td>{result['serious']:.0%}</td></tr></table>"
            )
            if q["dropped"]:
                res += "<p class='muted'>Not kept: " + e("; ".join(q["dropped"])) + "</p>"
        form = (
            "<form method='post' action='/intake'><textarea name='message'>" + e(message) + "</textarea><p><select "
            "name='source'>"
            + "".join(f"<option value='{s}'>{e(SOURCE_LABEL[s])}</option>" for s in SOURCES)
            + "</select> <button>Read it</button></p></form>"
        )
        return _page(
            "Read an inquiry",
            "<h1>Read an inquiry</h1><p class='muted'>Turns a message into what the person "
            "wants: Claude when an API key is set (each field must quote the message; areas we don't cover are "
            f"dropped), simple rules otherwise.</p><p class='muted'>{e(note)}</p>{form}{res}",
        )

    sample = "Hi, we're pre-approved up to $450k and need 3 beds in Riverside or Hillcrest, ideally within 6 weeks."

    @app.get("/intake", response_class=HTMLResponse)
    def intake() -> str:
        return intake_page(sample, score_new(c, LeadIn(message=sample)), "Example, read with rules.")

    @app.post("/intake", response_class=HTMLResponse)
    def intake_post(message: str = Form(""), source: str = Form("website")) -> str:
        if not message.strip():
            raise HTTPException(422, "paste an inquiry")
        return intake_page(
            message,
            score_new(c, LeadIn(message=message, source=source)),
            "Read with Claude." if c.reader is not None else "Read with rules (no API key set).",
        )

    return app


def demo_app() -> FastAPI:
    import os

    return create_app(demo_console(Reader() if os.environ.get("ANTHROPIC_API_KEY") else None))


__all__ = ["Console", "Inquiry", "create_app", "demo_app", "demo_console"]
