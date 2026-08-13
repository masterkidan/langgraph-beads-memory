# Confounded runs — kept for provenance, excluded from the tables

A run lands here when the two arms did not run the same experiment, so the
comparison measures something other than the memory layer. Excluded on stated
evidence, never because the result was unwelcome.

## `granite4_1_8b-vecdb` (2026-08-12)

The treatment delegated all three researchers on **conv-1**, a turn that only
states constraints. The baseline did not.

```
treatment conv-1:  remember_fact ×3, researcher_pgvector, researcher_qdrant, researcher_weaviate
baseline  conv-1:  manage_memory
```

The treatment therefore researched twice — once in conv-1, again in conv-2 —
which inflates its token count and corrupts the delegation the scenario claims
to measure. Its reported figures were −17 pts accuracy and +28% input tokens;
neither is usable.

`granite4.1:8b`'s **incident** pair is unaffected and stays in the study,
including its +44% token result.

## Earlier, same cause

`results/n1-vecdb-gemma-confounded/` — `gemma4:12b`, treatment delegated on
conv-1 (602s, three researchers) while the baseline took 33s and did not.

## The recurring cause

Delegation discipline is enforced by prompt wording, and that wording does not
transfer across models. It was tuned against one model's failure mode, then
had to be restructured for a second, and a third ignored it anyway. Any new
model should have the opening turn of each scenario traced for spurious
delegation before a real run — and a structural guard would be more reliable
than more prompt engineering.
