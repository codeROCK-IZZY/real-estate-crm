"""Inquiry reading through Claude (real SDK, canned responses), the rules fallback, the console and the CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from recrm.app import AREAS, create_app, demo_console
from recrm.cli import main
from recrm.intake import FALLBACK_BETA, MODEL, Reader, rules

MSG = "Hi, we're pre-approved up to $450k and need 3 beds in Riverside or Hillcrest, ideally within 6 weeks."


def message(*content: dict[str, Any], stop: str = "tool_use") -> dict[str, Any]:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": list(content),
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class Script:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses: Iterator[dict[str, Any]] = iter(responses)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx2.Headers] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        return httpx2.Response(200, json=next(self.responses))


def reader(script: Script) -> Reader:
    client = anthropic.Anthropic(
        api_key="test",
        base_url="http://api.test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(script)),
    )
    return Reader(client)


def call(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "goal": "buy",
        "budget_max": None,
        "bedrooms_min": None,
        "areas": [],
        "timeline_weeks": None,
        "pre_approved": None,
        "quotes": {"budget_max": "", "bedrooms_min": "", "areas": "", "timeline_weeks": "", "pre_approved": ""},
    }
    base.update(fields)
    return {"type": "tool_use", "id": "t", "name": "record_inquiry", "input": base}


def test_reader_request_shape_and_a_clean_read() -> None:
    script = Script(
        [
            message(
                call(
                    budget_max=450000,
                    bedrooms_min=3,
                    areas=["Riverside", "Hillcrest"],
                    timeline_weeks=6,
                    pre_approved=True,
                    quotes={
                        "budget_max": "up to $450k",
                        "bedrooms_min": "3 beds",
                        "areas": "Riverside or Hillcrest",
                        "timeline_weeks": "within 6 weeks",
                        "pre_approved": "we're pre-approved",
                    },
                )
            )
        ]
    )
    q = reader(script).read(MSG, AREAS)
    assert (q.goal, q.budget_max, q.bedrooms_min, q.areas, q.timeline_weeks, q.pre_approved) == (
        "buy",
        450000,
        3,
        ["Riverside", "Hillcrest"],
        6,
        True,
    )
    assert q.dropped == [] and q.source == "claude"
    req = script.requests[0]
    assert req["model"] == MODEL and req["output_config"] == {"effort": "low"} and req["fallbacks"] == "default"
    assert "thinking" not in req and "tool_choice" not in req
    assert req["tools"][0]["strict"] is True and "Riverside" in req["system"][0]["text"]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"} and "<inquiry>" in req["messages"][0]["content"]
    assert FALLBACK_BETA in script.headers[0]["anthropic-beta"]


def test_unsupported_fields_are_dropped_with_reasons() -> None:
    script = Script(
        [
            message(
                call(
                    budget_max=900000,
                    bedrooms_min=50,
                    areas=["Riverside", "Atlantis"],
                    pre_approved=True,
                    quotes={
                        "budget_max": "budget of 900k",
                        "bedrooms_min": "50 beds",
                        "areas": "Riverside",
                        "timeline_weeks": "",
                        "pre_approved": "",
                    },
                )
            )
        ]
    )
    q = reader(script).read(MSG, AREAS)
    assert q.budget_max is None and q.bedrooms_min is None and q.pre_approved is None
    assert q.areas == ["Riverside"]
    joined = " | ".join(q.dropped)
    assert "budget_max: no quote" in joined and "bedrooms_min: 50 is outside" in joined
    assert "Atlantis" in joined and "pre_approved: no quote" in joined


def test_refusal_falls_back_to_rules() -> None:
    q = reader(Script([message({"type": "text", "text": "no"}, stop="refusal")])).read(MSG, AREAS)
    assert q.source == "rules" and q.budget_max == 450000 and "Claude gave no answer" in q.dropped[-1]


@pytest.mark.parametrize(
    ("text", "goal", "budget", "beds", "weeks", "approved"),
    [
        (MSG, "buy", 450000, 3, 6, True),
        ("Looking to sell our 4 bedroom house next month", "sell", None, 4, 4, None),
        ("Any 2br rentals under $1,800 a month?", "rent", 1800, 2, None, None),
        ("Mortgage approval came through yesterday, budget $1.2m", "buy", 1200000, None, None, True),
    ],
)
def test_rules(
    text: str, goal: str, budget: int | None, beds: int | None, weeks: int | None, approved: bool | None
) -> None:
    q = rules(text, AREAS)
    assert (q.goal, q.budget_max, q.bedrooms_min, q.timeline_weeks, q.pre_approved) == (
        goal,
        budget,
        beds,
        weeks,
        approved,
    )


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(demo_console()))


def test_console_pages(client: TestClient) -> None:
    assert client.get("/healthz").json()["claude"] is False
    for path in ("/", "/pipeline", "/transactions", "/transactions/1", "/intake"):
        assert client.get(path).status_code == 200, path
    q = client.get("/api/queue?limit=5").json()
    assert len(q) == 5 and q[0]["priority"] >= q[-1]["priority"]
    assert client.get(f"/leads/{q[0]['lead']}").status_code == 200
    assert client.get("/leads/999999").status_code == 404 and client.get("/transactions/99").status_code == 404


def test_scoring_a_new_lead_and_deadlines_api(client: TestClient) -> None:
    r = client.post(
        "/api/leads", json={"message": MSG, "source": "referral", "views": 8, "saves": 3, "phone": True}
    ).json()
    weak = client.post("/api/leads", json={"message": "Just browsing for now.", "source": "social"}).json()
    assert r["inquiry"]["budget_max"] == 450000 and r["serious"] > weak["serious"]
    assert client.post("/api/leads", json={"message": "x", "source": "billboard"}).status_code == 422
    d = client.post("/api/transactions", json={"contract": "2026-11-20", "closing": "2026-12-28"}).json()
    assert d[0]["task"] == "Earnest money deposited" and d[-1]["due"] == "2026-12-28"
    assert client.post("/api/transactions", json={"contract": "2026-11-20", "closing": "2026-11-25"}).status_code == 422
    assert client.post("/intake", data={"message": MSG, "source": "website"}).status_code == 200
    assert client.post("/intake", data={"message": " "}).status_code == 422


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["deadlines", "2026-11-20", "2026-12-28"]) == 0
    assert "Earnest money" in capsys.readouterr().out
    assert main(["deadlines", "2026-11-20", "2026-11-22"]) == 2
    assert main(["read", "Selling a 3 bed in Old Town"]) == 0
    assert '"goal": "sell"' in capsys.readouterr().out
