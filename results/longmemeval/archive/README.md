# Archived LongMemEval runs — do not cite

**These runs are superseded. Nothing here should be quoted in the README, the
results write-ups, or an article.** They are kept because several of them are
the evidence behind a disclosed correction, not because their numbers stand.

The current benchmark is the three files in the parent directory:

```
budget1200-n78-autolink-gemma4_12b.json     tail vs augment, n=78
budget6000-n78-autolink-gemma4_12b.json     tail vs augment, n=78
budget1200-n78-docstores-gemma4_12b.json    PostgresStore + bm25, n=78
```

## Why each of these is archived

**The four-arm head-to-head** (`knowledge-update-4arm`) — the comparison that is
easiest to misread, and did mislead a review of this repo. It lets every arm
take whatever context it wants: the document store spends 2,473 tokens, the
fact graph 328. It therefore measures **budgets, not memory strategies**, and
the store's apparent 57-to-48 win is an artefact of being handed 7.5× more
context. The budget-matched run in the parent directory inverts it, 51 to 42.

Its one still-useful number is the `fullcontext` ceiling — 64/78 at 6,337
tokens — which is why the file is kept rather than deleted.

**The memory-arm variants** (`agentingest`, `autolink`, `dag`,
`knowledge-update-gemma4_12b`) — all measure the *facts-only* arm, with no
transcript at all. Nobody deploys memory that way; the shipped middleware sends
windowed raw messages **plus** an injected block. These informed the choice of
ingest mode and nothing else.

**`k16-substfix`** — stopped at 31 of 78. Incomplete, never analysed.

**The `num_ctx` sweep** (`ctx2048`, `ctx4096`, `ctx16384`) — manufactured
scarcity by shrinking the window rather than by lengthening the session, so it
handicaps *reasoning* as well as evidence for every arm. `ctx16384` also only
reached n=6. Superseded by the budget sweep, which holds the window fixed and
varies only the allowance.

**The n=25 budget runs** (`budget1200/3000/6000-gemma4_12b`) — the first version
of the context-budget curve, on `replay` ingest. Superseded at the ends by the
n=78 runs. The 6,000 point here showed **−2** and was described in an earlier
draft as measured interference; at n=78 it is −1 with a 4-to-5 pairwise split,
i.e. a tie. That correction is why this file is kept.

## Reading them anyway

If you do, the two things that make almost every number here incomparable with
the current set:

1. **Budgets differ between arms** unless the filename says `budget`.
2. Runs before 2026-08-19 were made while Ollama truncated prompts at its
   default `num_ctx` of 2048, so their token accounting is a floor and their
   accuracy was measured under an unannounced handicap.
