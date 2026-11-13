"""Which leads buy: logistic regression on what the brokerage can see, trained on past leads.

Training data is the brokerage's own history (a past period worked newest-first, the usual habit): every lead that
was reached on the first or second day, labelled by whether they booked a viewing. Leads reached late are left out, because by then whether they answered says more about
timing than about the lead. Fitted with numpy (L2-regularised gradient descent), no other dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .world import FEATURE_NAMES, Brokerage


@dataclass
class Model:
    weights: list[float] = field(default_factory=list)

    def predict(self, x: np.ndarray) -> float:
        return float(1 / (1 + np.exp(-float(np.dot(self.weights, x)))))

    def explain(self, x: np.ndarray) -> list[tuple[str, float]]:
        """Each signal's push on the log-odds, largest first (the bias left out)."""
        parts = [(FEATURE_NAMES[i], float(self.weights[i] * x[i])) for i in range(1, len(x)) if x[i]]
        return sorted(parts, key=lambda p: -abs(p[1]))

    def to_json(self) -> str:
        return json.dumps({"features": FEATURE_NAMES, "weights": [round(w, 5) for w in self.weights]})

    @classmethod
    def from_json(cls, text: str) -> Model:
        return cls(list(json.loads(text)["weights"]))


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 2000, lr: float = 0.3) -> Model:
    w = np.zeros(x.shape[1])
    n = len(y)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]] / n
        w -= lr * grad
    return Model([float(v) for v in w])


def history(brokers: list[Brokerage], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Past leads worked newest-first, kept when reached within two days of arriving."""
    from .queue import simulate_history

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for b in brokers:
        for x, y in simulate_history(b, seed=seed):
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)
