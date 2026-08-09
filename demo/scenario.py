"""The scripted 3-conversation scenario. Identical for both conditions; the only
variable is the memory layer."""

RESEARCH_SYSTEM_PROMPT = (
    "You are a research analyst. You have memory tools: when you reach a "
    "conclusion, record it with remember_fact. When the user revises an "
    "earlier constraint, record the new value with remember_fact using "
    "relation='supersedes' and the old fact's short id from your Memory "
    "context. Use the read_document tool to research; delegate sub-topics "
    "to your researcher sub-agents when asked to investigate in depth."
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
            # Buried-detail question: lives only in qdrant.md's memory section.
            "And remind me — what was that big memory optimization for the "
            "strongest runner-up, and roughly how much did it save?",
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
