"""Playground: one message, two memory layers, side by side.

The benchmarks answer "is it better" with numbers. This answers "what is it
actually doing" — you type once, both arms answer, and the treatment shows the
facts it captured and the ranked set it injected.

Each chat is one `session_id`, shared by both arms and spanning every message,
which is the property under demonstration: a new message is a new LangGraph
thread, so anything either side remembers had to come from its memory layer.

    uv run uvicorn playground.app:app --port 8100
"""

from __future__ import annotations

import pathlib
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.conditions import build
from demo.scenarios import SCENARIOS, Scenario
from playground.search import web_search

STATIC = pathlib.Path(__file__).parent / "static"

PLAYGROUND_PROMPT = (
    "You are a helpful research assistant with durable memory across "
    "conversations.\n\n"
    "- Use web_search when you need facts you do not already know.\n"
    "- When the user states a preference, constraint or correction, record it "
    "with remember_fact so it survives into later conversations.\n"
    "- When the user corrects something you were told earlier, record the new "
    "value with remember_fact using relation='supersedes' and the old fact's "
    "short id from your Memory context.\n"
    "- Answer directly. Do not narrate your tool use."
)


def _playground_scenario() -> Scenario:
    """Freeform chat with web search, no scripted turns and no sub-agents.

    Registered so `demo.conditions.build` can construct either arm with the
    same wiring the benchmarks use — the playground must exercise the real
    code path, not a simplified copy of it.
    """
    return Scenario(
        name="playground",
        root_prompt=PLAYGROUND_PROMPT,
        subagent_prompt="",
        subtopics=(),
        corpus_dir=STATIC,  # unused; no corpus reader in this scenario
        conversations=(),
        score=lambda transcript: {},
        tools_factory=lambda _s: [web_search],
    )


SCENARIOS["playground"] = _playground_scenario

app = FastAPI(title="beads-memory playground")
_chats: dict[str, dict] = {}


class NewChat(BaseModel):
    title: str | None = None


class Message(BaseModel):
    text: str


def _make_chat(title: str | None) -> dict:
    chat_id = uuid.uuid4().hex[:8]
    session_id = f"pg-{chat_id}"
    arms = {}
    for arm in ("baseline", "treatment"):
        # NOT "pg_" — Postgres reserves that prefix for system schemas.
        schema = f"play_{arm.replace('-', '_')}_{chat_id}"
        (invoke, cleanup), _ = build(arm, "playground", session_id, schema)
        arms[arm] = {"invoke": invoke, "cleanup": cleanup, "schema": schema}
    return {
        "id": chat_id,
        "title": title or "New chat",
        "session_id": session_id,
        "arms": arms,
        "turns": [],
        "created": time.time(),
    }


@app.post("/api/chats")
def create_chat(body: NewChat):
    chat = _make_chat(body.title)
    _chats[chat["id"]] = chat
    return {"id": chat["id"], "title": chat["title"], "session_id": chat["session_id"]}


@app.get("/api/chats")
def list_chats():
    return [
        {"id": c["id"], "title": c["title"], "turns": len(c["turns"])}
        for c in sorted(_chats.values(), key=lambda c: -c["created"])
    ]


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    return {"id": chat["id"], "title": chat["title"], "turns": chat["turns"]}


def _run_arm(chat: dict, arm: str, text: str, thread_id: str) -> dict:
    """One arm's answer, plus what its memory did. Never raises."""
    recorder: list = []
    started = time.monotonic()
    try:
        result = chat["arms"][arm]["invoke"](thread_id, text, recorder=recorder)
        msgs = result["messages"]
        answer = str(msgs[-1].content) if msgs else ""
        tokens = sum(
            (getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0) for m in msgs
        )
        tools = [tc.get("name") for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    except Exception as e:  # noqa: BLE001 — one arm failing must not lose the other
        return {
            "answer": "",
            "error": f"{type(e).__name__}: {e}",
            "seconds": round(time.monotonic() - started, 1),
            "input_tokens": 0,
            "tools": [],
            "injected": [],
        }
    # Only the treatment has an injection log: its recall is automatic, so there
    # is no tool call to observe. The baseline's recall IS a search_memory call
    # and shows up in `tools`.
    injected = recorder[0]["injected"] if recorder else []
    return {
        "answer": answer,
        "error": None,
        "seconds": round(time.monotonic() - started, 1),
        "input_tokens": tokens,
        "tools": tools,
        "injected": injected,
    }


@app.post("/api/chats/{chat_id}/messages")
def send(chat_id: str, body: Message):
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")

    # A NEW thread per message, deliberately. LangGraph's checkpointer keeps
    # history within a thread, so a fresh thread means neither arm can lean on
    # message history — anything recalled came from its memory layer.
    turn_no = len(chat["turns"])
    thread_id = f"{chat['session_id']}-t{turn_no}"

    # Sequential, not parallel: one Ollama serves both, so concurrency would
    # only make each slower and muddy the per-arm timings.
    out = {"user": text, "n": turn_no}
    for arm in ("baseline", "treatment"):
        out[arm] = _run_arm(chat, arm, text, thread_id)
    chat["turns"].append(out)
    if turn_no == 0 and chat["title"] == "New chat":
        chat["title"] = text[:48]
    return out


@app.get("/api/chats/{chat_id}/memory")
def memory(chat_id: str):
    """What the treatment has stored for this chat, by kind and source."""
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    from demo.harness import snapshot_memory

    return {
        "baseline": snapshot_memory(
            "baseline", chat["arms"]["baseline"]["schema"], chat["session_id"]
        ),
        "treatment": snapshot_memory(
            "treatment", chat["arms"]["treatment"]["schema"], chat["session_id"]
        ),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
