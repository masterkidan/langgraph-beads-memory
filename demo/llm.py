"""Model configuration for the demo. One model for agents AND judge."""

import os

from langchain_ollama import ChatOllama

MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen3:8b")


def make_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL, temperature=temperature)
