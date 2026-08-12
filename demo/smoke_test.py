"""Pre-flight: verify the local model can do what BOTH conditions need.
Failing this means 'pick a better model', not 'the memory layer lost'."""

import re
import sys

from langchain_core.messages import HumanMessage

from demo.llm import make_llm

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """qwen3 is a reasoning model and may wrap its scratch-work in
    <think>...</think> before the real answer. That's a formatting artifact,
    not a capability signal, so strip it before checking content."""
    return _THINK_BLOCK.sub("", text).strip()


def check_tool_calling() -> bool:
    """(b) treatment needs reliable tool calls with a resolvable fact reference."""
    from langchain_core.tools import tool

    @tool
    def remember_fact(body: str, relates_to: str | None = None, relation: str | None = None) -> str:
        """Durably remember a conclusion you have reached. body is the full text of
        the conclusion. Use relates_to (a single short fact id string like
        "fact-a3f8b2c1") with relation "supersedes" when this conclusion replaces
        an earlier fact."""
        return "Remembered [fact-12345678]"

    llm = make_llm().bind_tools([remember_fact])
    prompt = (
        "Context fact: [fact-aaaa1111] (user_input) the budget is 100k per year.\n"
        "The user just said the budget is now 50k, not 100k. Record this revision "
        "using the remember_fact tool."
    )
    resp = llm.invoke([HumanMessage(prompt)])
    ok = bool(resp.tool_calls)
    if ok:
        args = resp.tool_calls[0]["args"]
        # Use the SAME normalizer the real tools use, so this gate reflects
        # what production code actually accepts.
        from beads_memory.tools import _normalize_reference

        ref, rel_override = _normalize_reference(args.get("relates_to"))
        relation = args.get("relation") or rel_override
        ok = (
            "50k" in str(args.get("body", ""))
            and ref is not None
            and "aaaa1111" in ref
            and relation == "supersedes"
        )
    print(f"tool-calling + supersedes: {'PASS' if ok else 'FAIL'} ({resp.tool_calls})")
    return ok


def check_extraction_prereq() -> bool:
    """(a) baseline needs the model to make LangMem-style extraction work;
    proxy check: structured JSON extraction from a transcript."""
    llm = make_llm()
    resp = llm.invoke(
        [
            HumanMessage(
                "Extract durable facts as a JSON list of strings from: "
                '"My budget is 100k and I only trust primary sources." '
                "Answer with ONLY the JSON list."
            )
        ]
    )
    text = _strip_think(str(resp.content))
    low = text.lower()
    # Accept any surface form of the figure. A literal "100k" check FAILED
    # qwen3.5:9b for answering ["The user\'s budget is $100,000.", "The user
    # trusts only primary sources."] — a correct extraction of both facts.
    # This is a gate: a false FAIL excludes a capable model from the benchmark
    # entirely, which is a worse error than a false PASS. It is also the same
    # bug the objective metrics already had ("50k" vs "$50,000"), never
    # propagated here.
    budget = any(v in low for v in ("100k", "100,000", "100000"))
    sources = "primary source" in low
    listish = "[" in text
    ok = budget and sources and listish
    if not ok:
        missing = [
            n
            for n, v in (("budget", budget), ("primary-sources", sources), ("json-list", listish))
            if not v
        ]
        print(f"extraction-shaped output: FAIL (missing: {', '.join(missing)}) ({text[:160]})")
    else:
        print(f"extraction-shaped output: PASS ({text[:160]})")
    return ok


def check_embeddings() -> bool:
    from beads_memory.embeddings import OllamaEmbedder

    emb = OllamaEmbedder()
    v = emb.embed("hello world")
    ok = len(v) == 768
    print(f"nomic-embed-text 768-d: {'PASS' if ok else 'FAIL'} (len={len(v)})")
    return ok


if __name__ == "__main__":
    results = [check_tool_calling(), check_extraction_prereq(), check_embeddings()]
    sys.exit(0 if all(results) else 1)
