# LinkedIn post — draft

Continuing my exploration with agents — this time on what happens to an agent's memory over a long session.

The thing that started it: I watched a model write "I've recorded that correction in my memory: Deploy time 13:20 UTC" — and make no tool call at all. The store was empty. It then searched that empty store three times in later turns and correctly reported it had nothing.

In most agent memory setups, saving and recalling are two independent decisions the model has to make, and nothing reconciles them.

So I tried the other approach: capture in middleware rather than in a tool, and store a typed graph of individual claims instead of saved documents. Two things fall out of that, and they turned out to be the whole story:

→ Retrieval cost stops growing. It's k facts per call regardless of how much the session has accumulated — the store grew 12× across a run and the injected block didn't move.

→ The payload gets small, because a claim isn't a document. 793 characters against 3,653 for the same question. And because it lives in the system prompt rather than the message history, it's replaced each call instead of re-sent.

Measured on the same scenario and model: 29% fewer input tokens at equal accuracy.

Plenty of trial and error getting there — honestly, more of it in my measurement than in the library. One metric was scoring a *correct* answer as wrong because it matched "check" inside the word "checkout". A resilience policy I'd written and documented could never have worked. My own agent's restatements grew to 58% of everything stored, which is why the "efficient" memory layer was initially more expensive than the baseline.

All of it is written up, including the bits I got wrong and a set of predictions I committed to git before running — one of which I got backwards with high confidence.

[link]

---

## Notes for posting

- Lead is the anecdote, not the architecture — it's concrete and it's the actual reason the project exists.
- The trial-and-error paragraph is deliberately specific. Vague admissions of "learning" read as false modesty; named bugs read as a real account.
- No adjectives on the numbers. 29% is either interesting to the reader or it isn't.
- Second article (five models, and why "the better model wins" isn't the answer) is teased at the end of the Medium piece rather than here, to avoid promising two things at once.
