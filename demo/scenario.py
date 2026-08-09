"""The scripted 3-conversation scenario. Identical for both conditions; the only
variable is the memory layer."""

# Delegation discipline is spelled out because profiling showed the agent
# spawning all three researchers on conversation 1 — a turn that only states
# constraints and asks for nothing. That burned ~140s per researcher on work
# nobody requested, roughly a third of a run, and it also corrupted the
# measurement: conversation 2 was then re-researching rather than researching.
# Both conditions get the identical wording, so this does not favour either.
RESEARCH_SYSTEM_PROMPT = (
    "You are a research analyst. You have memory tools: when you reach a "
    "conclusion, record it with remember_fact. When the user revises an "
    "earlier constraint, record the new value with remember_fact using "
    "relation='supersedes' and the old fact's short id from your Memory "
    "context.\n\n"
    "Delegation rules — follow exactly:\n"
    "- Only delegate to your researcher sub-agents when the user explicitly "
    "asks you to investigate, research, or compare the options.\n"
    "- If the user is only stating requirements, giving you constraints, "
    "correcting an earlier detail, or asking a question you can answer from "
    "your Memory context, do NOT delegate and do NOT call read_document. "
    "Acknowledge briefly, record what matters with remember_fact, and stop.\n"
    "- Never delegate the same topic twice; if a researcher already reported "
    "on a topic, use that conclusion from your Memory context instead."
)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused researcher. Read the relevant documents thoroughly "
    "with read_document, record notable findings with remember_fact, and "
    "you MUST finish by calling conclude_task with a summary of your "
    "conclusion for your supervisor."
)

# conversation_id -> list of scripted user turns
CONVERSATIONS = [
    (
        "conv-1",
        [
            "We need to pick a vector database for our product. Constraints: "
            "the budget is $100k per year, it must be self-hostable, and I only "
            "trust primary benchmark data we measured ourselves, not vendor "
            "marketing numbers.",
        ],
    ),
    (
        "conv-2",
        [
            "Please investigate our vector database options in depth. Delegate "
            "pgvector, Qdrant, and Weaviate to your researchers, then give me "
            "your synthesis.",
            # The revision beat: supersedes gets exercised here.
            "One correction before we wrap up: the budget is $50k per year, " "not $100k.",
        ],
    ),
    (
        "conv-3",
        [
            "Given everything we've established, which vector database should "
            "we pick and why? Be specific about how it fits our constraints.",
            # Buried-detail question: this fact lives only in qdrant.md's memory
            # section, and reaches conversation 3 only via the Qdrant researcher's
            # rollup — which is exactly what we want to measure.
            #
            # DISCLOSED SCENARIO ITERATION (2026-08-08, after run 0; demo design
            # spec §8 permits iterating the scenario and requires saying so):
            # this question originally asked about "the strongest runner-up"
            # without naming it. That was ambiguous by accident — the referent
            # depends on which database the agent picked. In run 0 the treatment
            # picked Qdrant, read "runner-up" as Weaviate, and hallucinated an
            # optimization for it; the baseline scored the point only because it
            # declined to pick anything, leaving Qdrant as the default reading.
            # The question therefore measured which DB was chosen, not whether
            # the researcher's finding survived. Naming Qdrant removes the
            # ambiguity. This change was made AFTER seeing results and is
            # recorded here so the results write-up can state it plainly.
            "And remind me — what was that big memory optimization the Qdrant "
            "researcher found, and roughly how much did it save?",
        ],
    ),
]

# Ground truth for objective metrics (metrics.py checks these substrings).
# Terms with multiple plausible surface forms are lists of accepted variants —
# a model writing "32 times" instead of "32x" is correct and must score as such.
PLANTED = {
    "revised_budget_variants": ["50k", "50,000", "50000"],
    # must NOT be presented as current
    "stale_budget_variants": ["100k", "100,000", "100000"],
    "selfhost_variants": ["self-host", "self host", "selfhost", "on-prem", "on prem"],
    "constraint_primary_sources": "primary",
    "buried_detail_variants": [
        ["binary quantization"],
        ["32x", "32 times", "32-fold", "32 fold", "32×"],
    ],
    # $50k budget: pgvector ($12-18k) fits; qdrant ($30k) fits; weaviate ($60k) does not.
    "expected_pick_one_of": ["pgvector", "qdrant"],
}
