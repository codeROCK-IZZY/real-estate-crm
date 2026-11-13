# Known limitations

- **The evaluation is a simulation, and partly circular.** Seriousness in the simulated brokerages depends on the
  same kinds of signals the model reads (source, words, engagement, phone). The results show the method works when
  such signals exist and shifts are moderate, not how much it would lift a real brokerage. Real lead histories would
  be needed to measure that.
- **Calls are modelled simply.** A call takes 15 minutes and either connects or not. There are no texts or emails,
  no time zones, no agent specialisms, and no lead assignment beyond "the next free agent takes the top of the
  queue".
- **The model is trained on history worked newest-first,** so it learns from the leads that habit reached. Leads it
  never reached quickly are under-represented. Leaving out leads reached after two days limits the bias but doesn't
  remove it.
- **Retraining needs history.** A new brokerage starts with the shared model, which lost some ground in the shifted
  market (17% instead of 18%).
- **Deadlines use US federal holidays only.** State holidays, contract-specific counting rules and times of day are
  not configured in the demo (ADR 2).
- **Inquiry reading with Claude is not measured for accuracy** (ADR 3).
- **The console is a demo:** one brokerage held in memory at a fixed time, with no logins, no editing and no
  calendar or email sync.
