# Real estate CRM: evaluation

Simulated brokerages over two months: more leads than calling time, leads that cool at different speeds by source, and a hidden seriousness the CRM has to infer from what it can see.

## Tuning brokerages: 6 brokerages, 39 agents, 32,840 leads

| Who is called first | Deals | Viewings | Serious leads never called | Median wait for a serious lead |
|---|---:|---:|---:|---:|
| Oldest first (inbox order) | 152 | 391 | 4,418 of 6,234 | 334.5 h |
| Newest first (speed to lead) | 2,190 | 5,133 | 2,805 of 6,234 | 1.6 h |
| Referrals, then urgent words, then newest | 2,486 | 5,642 | 1,788 of 6,234 | 2.7 h |
| Predicted seriousness x reachability (this CRM) | 2,863 | 6,454 | 756 of 6,234 | 3.2 h |

## Held-out brokerages: 15 brokerages, 72 agents, 60,170 leads

| Who is called first | Deals | Viewings | Serious leads never called | Median wait for a serious lead |
|---|---:|---:|---:|---:|
| Oldest first (inbox order) | 370 | 741 | 7,966 of 11,236 | 334.5 h |
| Newest first (speed to lead) | 4,056 | 9,354 | 4,985 of 11,236 | 1.6 h |
| Referrals, then urgent words, then newest | 4,537 | 10,375 | 3,164 of 11,236 | 2.5 h |
| Predicted seriousness x reachability (this CRM) | 5,200 | 11,790 | 1,330 of 11,236 | 3.1 h |

## A shifted market (held back): 8 brokerages, 42 agents, 35,485 leads

| Who is called first | Deals | Viewings | Serious leads never called | Median wait for a serious lead |
|---|---:|---:|---:|---:|
| Oldest first (inbox order) | 163 | 362 | 6,391 of 8,986 | 334.5 h |
| Newest first (speed to lead) | 2,772 | 6,294 | 4,041 of 8,986 | 1.8 h |
| Referrals, then urgent words, then newest | 2,807 | 6,360 | 3,728 of 8,986 | 2.2 h |
| Predicted seriousness x reachability (this CRM) | 3,239 | 7,357 | 2,045 of 8,986 | 3.0 h |
| The same, retrained on the brokerage's own history | 3,274 | 7,545 | 1,536 of 8,986 | 3.0 h |

## Model

Cooling time per source, learned from call history (hours for reachability to fall by e): portal 19.6, website 37.8, open_house 60.4, social 16.3, referral 203.7, past_client 325.3

## Gate (regression limits, not targets)

- PASS heldout_deals_vs_newest_min: 1.282 (limit 1.1)
- PASS heldout_deals_vs_rules_min: 1.146 (limit 1.0)
- PASS shifted_deals_vs_newest_min: 1.168 (limit 0.95)
- PASS shifted_retrained_vs_newest_min: 1.181 (limit 1.0)
