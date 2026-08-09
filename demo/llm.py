"""Model configuration for the demo. One model for agents AND judge."""

import os

from langchain_ollama import ChatOllama

MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen3:8b")


# A hung request must fail, not stall forever. ChatOllama has no default
# timeout, and an N=3 run once sat for an hour with both the client and Ollama
# idle and five connections open — no error, no progress, nothing in the log.
# The harness records a failed turn and continues, so a timeout costs one turn;
# no timeout costs the whole run. Generous enough that a slow-but-healthy call
# on this hardware (~35s typical, long contexts slower) is never cut off.
REQUEST_TIMEOUT_S = float(os.environ.get("BEADS_DEMO_TIMEOUT", "300"))


def make_llm(temperature: float = 0.0, reasoning: bool | None = False) -> ChatOllama:
    """Build the demo LLM.

    `reasoning` defaults to False. qwen3 is a reasoning model, and with its
    default thinking mode ON it was measured to silently drop facts during
    extraction-style prompts (it returned only 1 of 2 facts, reproducibly, at
    temperature 0 across 3 runs; 773 reasoning tokens spent, second fact lost).
    That matters for fairness, not just quality: the LangMem baseline depends on
    extraction, so leaving thinking on would hobble the baseline for a reason
    that has nothing to do with the memory architecture under test — an
    accidental strawman. Disabling it fixed the same prompt in 18 tokens.

    Pass `reasoning=True` to opt back in for a specific call site.
    """
    return ChatOllama(
        model=MODEL,
        temperature=temperature,
        reasoning=reasoning,
        client_kwargs={"timeout": REQUEST_TIMEOUT_S},
    )
