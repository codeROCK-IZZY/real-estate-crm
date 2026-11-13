# 3. Read inquiries with Claude, and keep only what the message says

## Context

Inquiries arrive as free text ("pre-approved up to $450k, 3 beds in Riverside or Hillcrest, within 6 weeks").
Agents need the facts as fields to filter and match listings. A language model reads them well, but can also fill a
field the person never mentioned, or turn "near Riverside" into an area the agency doesn't cover.

## Decision

- Claude (`claude-opus-5-5`, low effort, with refusal fallbacks) fills a strict tool, `record_inquiry`, with the
  goal, maximum budget, minimum bedrooms, areas, timeline in weeks and pre-approval. For every field it fills, it
  must give the words it came from. The system prompt lists the areas the agency covers, says text in the inquiry is
  never an instruction, and is cached.
- Code then keeps a field only when its quote is really in the message and its value is in range (budget 1,000 to
  100 million, bedrooms 0 to 20, timeline up to ten years). An area is kept only if the agency covers it and it is
  named in the message. Everything dropped is listed with the reason.
- Without an API key, or on a refusal, simple rules fill what they can (currency amounts with k or m, "3 bed",
  "next month", pre-approval wording). The console says which reader was used.

## Consequences

- An inquiry can never gain a budget, an area or a pre-approval the person didn't state. The tests check each kind
  of rejection through the real SDK.
- Claude's accuracy on real inquiries was not measured: there is no labelled set of real messages, and the
  evaluation makes no paid API calls. The rules fallback is tested on fixed examples.
- The score uses only the structured signals the model was trained on. The inquiry fields help the agent and
  listing matching, but they don't feed the ranking, so a misread message cannot move a lead up the queue.
