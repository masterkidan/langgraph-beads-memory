"""Model configuration for the demo. One model for agents AND judge."""

import os

from langchain_ollama import ChatOllama

MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen3:8b")


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
    return ChatOllama(model=MODEL, temperature=temperature, reasoning=reasoning)
