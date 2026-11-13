"""Contract-to-close deadlines.

A purchase contract sets its deadlines as a number of days from the contract (or "effective") date: earnest money
within 3 business days, the inspection contingency in 10 days, the appraisal in 17, the loan commitment in 21, and so
on, and a closing date. Whether a term counts business days or calendar days depends on the contract; a calendar
deadline that lands on a weekend or a public holiday moves to the next business day, as most standard contracts say.
Missing one can let the other side walk away, so each becomes a task with a due date and a status.

Holidays are US federal holidays with their observed dates (a Saturday holiday is observed on Friday, a Sunday one on
Monday). Other calendars can be passed in.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d


def us_holidays(year: int) -> set[dt.date]:
    """US federal holidays (observed dates) for a year."""
    fixed = [
        dt.date(year, 1, 1),
        dt.date(year, 6, 19),
        dt.date(year, 7, 4),
        dt.date(year, 11, 11),
        dt.date(year, 12, 25),
    ]
    moving = [
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _nth_weekday(year, 11, 3, 4),
    ]
    days = {_observed(d) for d in fixed} | set(moving)
    nxt = _observed(dt.date(year + 1, 1, 1))
    if nxt.year == year:  # New Year's Day on a Saturday is observed on 31 December
        days.add(nxt)
    return days


@dataclass
class Calendar:
    holidays: set[dt.date] = field(default_factory=set)

    @classmethod
    def us(cls, *years: int) -> Calendar:
        days: set[dt.date] = set()
        for y in years:
            days |= us_holidays(y)
        return cls(days)

    def business(self, d: dt.date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def roll(self, d: dt.date) -> dt.date:
        while not self.business(d):
            d += dt.timedelta(days=1)
        return d

    def add_business(self, d: dt.date, n: int) -> dt.date:
        while n > 0:
            d += dt.timedelta(days=1)
            n -= int(self.business(d))
        return d


@dataclass(frozen=True)
class Term:
    key: str
    label: str
    days: int
    business: bool  # count business days (else calendar days, rolled forward off weekends and holidays)


DEFAULT_TERMS = (
    Term("earnest", "Earnest money deposited", 3, True),
    Term("inspection", "Inspection contingency ends", 10, False),
    Term("appraisal", "Appraisal contingency ends", 17, False),
    Term("financing", "Loan commitment due", 21, False),
)


@dataclass
class Task:
    key: str
    label: str
    due: dt.date
    done: bool = False

    def status(self, today: dt.date, cal: Calendar) -> str:
        if self.done:
            return "done"
        if today > self.due:
            return "overdue"
        if cal.add_business(today, 2) >= self.due:
            return "due soon"
        return "upcoming"


def schedule(
    contract: dt.date, closing: dt.date, terms: tuple[Term, ...] = DEFAULT_TERMS, cal: Calendar | None = None
) -> list[Task]:
    """Every deadline for a contract, in date order. The walk-through is the business day before closing."""
    cal = cal or Calendar.us(contract.year, contract.year + 1)
    tasks = []
    for t in terms:
        due = cal.add_business(contract, t.days) if t.business else cal.roll(contract + dt.timedelta(days=t.days))
        tasks.append(Task(t.key, t.label, due))
    close = cal.roll(closing)
    walk = close - dt.timedelta(days=1)
    while not cal.business(walk):
        walk -= dt.timedelta(days=1)
    tasks += [Task("walkthrough", "Final walk-through", walk), Task("closing", "Closing", close)]
    problems = [x for x in tasks if x.key != "closing" and x.due > close]
    if problems:
        raise ValueError("these deadlines fall after closing: " + ", ".join(x.label for x in problems))
    return sorted(tasks, key=lambda x: (x.due, x.key))
