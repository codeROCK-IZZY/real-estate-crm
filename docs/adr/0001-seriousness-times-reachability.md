# 1. Rank the call queue by seriousness times reachability

## Context

A brokerage gets more inquiries than its agents have calling time. Most CRMs show the inbox oldest first. The usual
advice ("speed to lead") is to call the newest inquiry at once. Neither asks the two questions that decide whether a
call is worth making now: is this person likely to be serious, and will they still answer?

## Decision

Priority = P(serious) × e^(−age / cooling time of the lead's source).

- **P(serious)** comes from a logistic regression on what the brokerage can see: source, urgent or casual words in
  the inquiry, budget against the area's prices, listings viewed and saved, and whether a phone number was given.
  It is trained on a past period worked newest-first, using the leads reached within two days, labelled by whether
  they booked a viewing. Leads reached later are left out, because by then whether they answered says more about
  timing than about the lead.
- **Cooling time per source** is estimated from the brokerage's past calls by maximum likelihood: which calls were
  answered at which lead age. It is never read from the simulation's own settings. The estimates (portal 20 h,
  social 16 h, website 38 h, referral 204 h) come close to the truth anyway.
- One setting was tuned on the 6 tuning brokerages: an extra pull towards fast-cooling sources (α). The values 0,
  0.5 and 1 gave 2,863, 2,602 and 2,099 deals. α = 0 was kept, because the reachability term already does that job.

**A change after the first held-out run.** The first version trained on a history worked oldest-first. The lead page
then showed urgent words ("pre-approved and ready to move") *lowering* a lead's score. Oldest-first almost never
reaches a lead within two days, so the model had seen only a few hundred examples, and it got the signs of the word
and budget signals wrong. Training now uses a history worked newest-first (16,064 examples), and the signs come out
as expected. α was checked again on the tuning brokerages, and every bank was run again. The first run had 4,988
held-out deals (+23% over newest first), 2,170 serious leads never called, and +12% and +14% in the shifted market.

## Consequences

- Held-out brokerages (after the change above): 5,200 deals against 4,056 for newest first (+28%) and 4,537 for
  rules (+15%), with serious leads never called down from 4,985 to 1,330.
- The median serious lead waits 3.1 hours instead of 1.6, deliberately: a referral can wait until the afternoon and
  a serious portal lead cannot.
- In a shifted market the trained model still won (+17%), and retraining on the brokerage's own history recovered a
  little more (+18%). So the CRM retrains per brokerage, and the evaluation gate checks both.
- Every score can be explained: the console lists each signal's push on the odds.
