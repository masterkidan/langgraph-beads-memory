"""Model configuration for the demo. One model for agents AND judge."""

import os

from langchain_ollama import ChatOllama

# qwen3:4b, not 8b. On a 16 GB machine 8b needs ~5.2 GB of *wired* Metal
# memory, which can neither swap nor compress; with ~0.1 GB free the model was
# evicted and reloaded repeatedly, turning a 6-second generation into 35 seconds
# and a delegation turn into an apparent hang. 4b needs ~2.5 GB and fits.
#
# It is not a downgrade for this workload: 4b passed the same pre-flight gate and
# emitted a *better-formed* tool call than 8b, which reliably nests the fact
# reference into a dict (the reason tools.py needs a coercion layer at all).
MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen3:4b")


# CORRECTION: an earlier comment here claimed client-side timeouts do not work.
# That was wrong — the test behind it was faulty. Verified against a socket that
# accepts and never replies, client_kwargs={"timeout": 5} raises ReadTimeout in
# 5.0s exactly.
#
# It is still not sufficient on its own. httpx's is a *per-read* timeout, so a
# response that trickles bytes resets it indefinitely — which is how a real run
# hung for 8.5 minutes with this set to 300s. The harness therefore also applies
# a hard per-turn deadline; see demo/harness.py.
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
