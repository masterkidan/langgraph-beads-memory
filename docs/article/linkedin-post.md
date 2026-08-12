# LinkedIn post — draft

Agent memory usually stores documents and retrieves documents. That works until sessions get long, and then two things start to cost.

**Save and recall are independent decisions.** In LangGraph's built-in setup, `manage_memory` and `search_memory` are both tools the model has to choose to call, and nothing reconciles them. In one benchmark run a model wrote "I've recorded that correction in my memory: Deploy time 13:20 UTC" — with no tool call. It then searched that empty store three times and correctly reported it had nothing.

**Recall arrives as a message, and messages accumulate.** A search result is a tool message: it sits in the history and is re-sent on every later model call in the same turn.

An alternative: capture in middleware rather than through a tool, and store a typed graph of individual claims instead of saved documents.

→ **Retrieval cost stops growing.** k facts per call, whatever the store holds. Across two scenarios the store grew 10–12× and the injected block did not move.

→ **The payload is small, because a claim is not a document.** 793 characters against 3,653 for the same question — and it lives in the system prompt, rewritten each call, rather than in the history where it stacks.

→ **Relevance comes from types.** Questions are held out of retrieval (they rank highly against a query precisely by resembling it — in one run four of eight slots were fragments of the question being asked). Superseded values are retired but kept for audit. Sub-agent findings are demoted, not deleted.

Measured on the same scenario and model: **29% fewer input tokens at equal accuracy**, while storing 5× more text.

Library, benchmark harness, both scenarios, and the full method — including the injection logs, so the ranking can be read off any run rather than taken on trust:

[link]

---

## Notes for posting

- Opens on the mechanism, not a personal account. The failed-tool-call example is evidence for the design argument, not an anecdote about the author.
- Three arrows map to the three README properties: constant, small, typed.
- No adjectives on the numbers.
- The model-comparison work is a separate post; not teased here to avoid promising two things at once.
