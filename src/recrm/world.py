"""Simulated brokerages: leads arriving over two months, agents with limited calling time, and buyers who cool off.

Each lead has a hidden intent (how likely they are to buy in the next few months) and a source: property portal,
the brokerage's website, an open-house sign-in, social media, a referral, or a past client. What the brokerage sees is
the source, the inquiry message, the budget against the area's prices, how many listings they viewed and saved, and
whether they left a phone number.

A lead is easiest to reach soon after they inquire: portal and social leads have usually asked several agents and
cool within a day, while referrals and past clients wait for days. Every call takes an agent 15 minutes; a lead that
answers and is serious books a viewing (another 30 minutes), and some viewings become closed deals.

The mapping from what is visible to intent is drawn per bank, so a `shifted` bank can change which sources and words
mean business.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOURCES = ("portal", "website", "open_house", "social", "referral", "past_client")
COOL_HOURS = {
    "portal": 20.0,
    "website": 40.0,
    "open_house": 60.0,
    "social": 16.0,
    "referral": 200.0,
    "past_client": 300.0,
}
HOT = [
    "We're pre-approved and ready to move.",
    "Can we see it this weekend?",
    "We need to move by next month.",
    "Our lease ends soon, looking to buy quickly.",
    "Mortgage approval came through yesterday.",
]
WARM = [
    "Is this still available?",
    "What are the property taxes?",
    "Does it have a garage?",
    "Could you send more photos?",
    "How old is the roof?",
]
COLD = [
    "Just browsing for now.",
    "What's the price?",
    "Thinking about buying next year.",
    "Curious what homes go for around here.",
    "Not sure we're ready yet.",
]


@dataclass
class Lead:
    id: int
    arrive: float  # hours since the start
    source: str
    message: str
    budget_ratio: float  # budget / area median price
    views: int
    saves: int
    phone: bool
    intent: float  # hidden: probability they buy if worked well
    # state
    first_contact: float | None = None
    attempts: int = 0
    reached: bool = False
    viewing: bool = False
    closed: bool = False
    lost: bool = False


@dataclass(frozen=True)
class Shape:
    """How visible signals relate to intent in a bank."""

    source_effect: tuple[float, ...] = (-0.6, 0.0, -0.3, -0.9, 1.0, 0.8)
    message_effect: float = 1.0  # strength of the hot/cold words
    engagement_effect: float = 0.35
    leads_per_agent_day: float = 14.0


@dataclass
class Brokerage:
    seed: int
    shape: Shape = field(default_factory=Shape)
    days: int = 60
    agents: int = 0
    leads: list[Lead] = field(default_factory=list)

    def __post_init__(self) -> None:
        g = np.random.default_rng(self.seed)
        self.agents = self.agents or int(g.integers(3, 9))
        n = int(self.shape.leads_per_agent_day * self.agents * self.days * float(g.uniform(0.8, 1.2)))
        mix = g.dirichlet(np.array([6.0, 3.0, 1.5, 3.0, 1.0, 1.0]) * 3)
        for i in range(n):
            day = float(g.uniform(0, self.days))
            hour = float(np.clip(g.normal(14, 4.5), 0, 23.99))  # inquiries peak in the afternoon and evening
            src_i = int(g.choice(len(SOURCES), p=mix))
            src = SOURCES[src_i]
            latent = float(g.normal(0, 1))
            z = self.shape.source_effect[src_i] + 0.9 * latent - 0.6
            word = float(g.random())
            p_hot = 1 / (1 + np.exp(-(latent * 1.2 * self.shape.message_effect - 0.8)))
            p_cold = 1 / (1 + np.exp(-(-latent * 1.2 * self.shape.message_effect - 0.6)))
            if word < p_hot * 0.6:
                msg = HOT[int(g.integers(len(HOT)))]
            elif word > 1 - p_cold * 0.6:
                msg = COLD[int(g.integers(len(COLD)))]
            else:
                msg = WARM[int(g.integers(len(WARM)))]
            views = int(g.poisson(np.exp(0.8 + self.shape.engagement_effect * latent)))
            saves = int(g.poisson(np.exp(-0.7 + self.shape.engagement_effect * latent)))
            budget = float(np.exp(g.normal(0.05 * latent, 0.35)))
            phone = bool(g.random() < 0.55 + 0.1 * np.tanh(latent))
            intent = float(1 / (1 + np.exp(-z)))
            self.leads.append(Lead(i, day * 24 + hour, src, msg, budget, views, saves, phone, intent))
        self.leads.sort(key=lambda lead: lead.arrive)
        for j, lead in enumerate(self.leads):
            lead.id = j


def features(lead: Lead) -> np.ndarray:
    """What the brokerage can see about a lead, as numbers."""
    m = lead.message
    return np.array(
        [1.0]
        + [1.0 if lead.source == s else 0.0 for s in SOURCES[1:]]
        + [
            1.0 if m in HOT else 0.0,
            1.0 if m in COLD else 0.0,
            float(np.log1p(lead.views)),
            float(np.log1p(lead.saves)),
            float(np.log(lead.budget_ratio)),
            1.0 if lead.phone else 0.0,
        ]
    )


FEATURE_NAMES = [
    "bias",
    *[f"source:{s}" for s in SOURCES[1:]],
    "hot words",
    "cold words",
    "log views",
    "log saves",
    "log budget ratio",
    "phone",
]
