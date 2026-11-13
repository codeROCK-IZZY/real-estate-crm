# 2. Work out contract deadlines in business days, with observed holidays

## Context

A purchase contract sets its deadlines relative to the contract date: "earnest money within 3 business days",
"inspection contingency 10 days", "loan commitment 21 days", plus a closing date. Agents usually count these by hand
on a calendar. The usual mistakes are counting a weekend or a holiday as a business day, forgetting that a deadline
on a Sunday moves to Monday, and missing that a holiday is observed on a Friday or Monday. A missed contingency can
let the other side walk away.

## Decision

- Each term is a number of days and a flag for business or calendar days. Business days skip weekends and holidays.
  A calendar-day deadline that lands on a weekend or holiday rolls forward to the next business day, as most
  standard contracts say.
- Holidays are US federal holidays with their observed dates. A Saturday holiday is observed on the Friday, and a
  Sunday one on the Monday, including New Year's Day observed on 31 December of the year before. Another calendar
  can be passed in.
- Closing rolls forward to a business day, and the final walk-through is the business day before closing.
- A schedule where any deadline falls after closing is refused with the names of the deadlines, rather than producing
  an impossible plan.
- Each deadline becomes a task with a status: done, overdue, due soon (within two business days) or upcoming.

## Consequences

- Tests pin the awkward cases: earnest money across Thanksgiving, a calendar deadline landing on a Sunday, a
  Saturday closing, the walk-through before a Christmas closing, and 4 July 2026 observed on Friday 3 July.
- State and contract variations (other holidays, deadlines counted from acceptance rather than the effective date,
  times of day) need their own terms and calendars, which the API accepts but the demo does not configure.
