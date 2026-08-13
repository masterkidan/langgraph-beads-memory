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

import contextlib
import pathlib
import threading
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
# Chat titles and transcripts survive a restart. The memory itself already
# lives in Postgres; losing the index to it on every reload was needless,
# especially when one message costs minutes of local inference.
STORE = pathlib.Path(__file__).parent / ".chats.json"

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

# One model call at a time, across every chat. A single Ollama serves both arms
# and every chat, so running them concurrently only makes each slower.
_gpu = threading.Lock()


def _save() -> None:
    """Persist everything except the live arm handles, which are rebuilt lazily."""
    import json

    data = {}
    for cid, c in _chats.items():
        rec = {k: v for k, v in c.items() if k != "arms"}
        # A chat restored from disk has arms=None until first use, and a turn is
        # saved before either arm is built. Read schemas from the live arms only
        # when they exist; otherwise the record already carries them.
        if c.get("arms"):
            rec["schemas"] = {a: c["arms"][a]["schema"] for a in c["arms"]}
        data[cid] = rec
    with contextlib.suppress(Exception):  # a failed save must never fail a turn
        STORE.write_text(json.dumps(data))


def _load() -> None:
    import json

    if not STORE.exists():
        return
    with contextlib.suppress(Exception):
        for cid, c in json.loads(STORE.read_text()).items():
            c["arms"] = None  # rebuilt on first use, see _arms
            _chats[cid] = c


def _arms(chat: dict) -> dict:
    """Build the two arms on demand, so a restored chat costs nothing until used."""
    if chat.get("arms"):
        return chat["arms"]
    arms = {}
    for arm in ("baseline", "treatment"):
        schema = (chat.get("schemas") or {}).get(arm) or f"play_{arm}_{chat['id']}"
        (invoke, cleanup), _ = build(arm, "playground", chat["session_id"], schema)
        arms[arm] = {"invoke": invoke, "cleanup": cleanup, "schema": schema}
    chat["arms"] = arms
    return arms


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
    _save()
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
        result = _arms(chat)[arm]["invoke"](thread_id, text, recorder=recorder)
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


class Rename(BaseModel):
    title: str


@app.patch("/api/chats/{chat_id}")
def rename(chat_id: str, body: Rename):
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    chat["title"] = body.title.strip()[:64] or chat["title"]
    chat["named"] = True
    _save()
    return {"id": chat_id, "title": chat["title"]}


@app.post("/api/chats/{chat_id}/turns")
def start_turn(chat_id: str, body: Message):
    """Register the user's message and return its index.

    Turns are created before either arm runs so the UI can show the message
    immediately and fill each answer in as it arrives. Both arms take minutes
    on local hardware; waiting for both before showing anything reads as a
    hang.
    """
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    n = len(chat["turns"])
    chat["turns"].append({"user": text, "n": n, "baseline": None, "treatment": None})
    if n == 0 and not chat.get("named"):
        chat["title"] = text[:48]
    _save()
    _schedule(chat, n)
    return {"n": n, "user": text, "title": chat["title"]}


def _execute_turn(chat: dict, n: int) -> None:
    """Run both arms for a turn, in the background.

    Execution used to be driven by the browser: the client POSTed once per arm.
    Anything that interrupted that loop — a reload, a closed tab — left the turn
    stranded forever, showing "queued" with nothing able to advance it. The
    server now owns execution, so a turn completes whether or not anyone is
    watching, and the UI is free to be a view over state rather than the thing
    that causes it.
    """
    turn = chat["turns"][n]
    thread_id = f"{chat['session_id']}-t{n}"
    for arm in ("baseline", "treatment"):
        if turn.get(arm):
            continue  # already done, e.g. resuming a partly-finished turn
        with _gpu:
            turn[f"{arm}_status"] = "running"
            turn[f"{arm}_started"] = time.time()
            _save()
            turn[arm] = _run_arm(chat, arm, turn["user"], thread_id)
            turn[f"{arm}_status"] = "done"
            _save()


def _schedule(chat: dict, n: int) -> None:
    for arm in ("baseline", "treatment"):
        chat["turns"][n].setdefault(f"{arm}_status", "queued")
    threading.Thread(target=_execute_turn, args=(chat, n), daemon=True).start()


def _resume_unfinished() -> None:
    """Re-queue anything left incomplete by a restart."""
    for chat in _chats.values():
        for turn in chat.get("turns", []):
            if not (turn.get("baseline") and turn.get("treatment")):
                _schedule(chat, turn["n"])


@app.get("/api/chats/{chat_id}/memory")
def memory(chat_id: str):
    """What the treatment has stored for this chat, by kind and source."""
    chat = _chats.get(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")
    from demo.harness import snapshot_memory

    return {
        "baseline": snapshot_memory(
            "baseline", _arms(chat)["baseline"]["schema"], chat["session_id"]
        ),
        "treatment": snapshot_memory(
            "treatment", _arms(chat)["treatment"]["schema"], chat["session_id"]
        ),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


_load()
_resume_unfinished()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
