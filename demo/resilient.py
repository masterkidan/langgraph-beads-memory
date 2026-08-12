"""Timeout, health-probe and recovery policy for a local Ollama server.

Ollama wedges: it accepts a TCP connection and never processes the request. The
runner unloads, `/api/version` and `/api/ps` keep answering 200, and the client
blocks in `sock_recv` indefinitely. Confirmed live — three ESTABLISHED
connections from the harness, nothing generating, and a *fresh* probe also
returning 000. This is ollama#15950.

Three layered timeouts existed before this (httpx 300s, a 900s turn deadline,
a 1020s watchdog) but no retry and no recovery, so a wedge cost 5-17 minutes to
notice and was never repaired. A bare retry is useless here — it dispatches onto
the same wedged server, often onto the same pooled httpx connection.

The policy that works is the one we were running by hand: detect, restart the
daemon, retry once with a fresh client. `probe()` deliberately uses
`/api/generate` rather than `/api/version`, because the control endpoints stay
healthy while generation is dead — watching the wrong endpoint is what made this
look like slowness for hours.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request

# Read the env var directly rather than importing demo.llm, which imports this
# module. Must track demo.llm.MODEL: probing a model that is not installed
# reports a healthy server as wedged, and the recovery path would then restart
# a daemon that was fine.
_DEFAULT_MODEL = os.environ.get("BEADS_DEMO_MODEL", "gemma4:12b")

CHAT_TIMEOUT_S = 120.0  # generous: healthy calls are 6-40s
PROBE_TIMEOUT_S = 45.0
RESTART_SETTLE_S = 8.0
PROBE_ATTEMPTS = 15


def probe(model: str = _DEFAULT_MODEL, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """True if the server will actually *generate*, not merely respond.

    Control endpoints answer 200 on a wedged server, so they cannot be used to
    decide health.
    """
    payload = (
        b'{"model":"' + model.encode() + b'","prompt":"ok","stream":false,'
        b'"options":{"num_predict":3}}'
    )
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def restart_and_wait(model: str = _DEFAULT_MODEL) -> bool:
    """Restart the daemon and poll until it generates. Returns False if it never does."""
    subprocess.run(
        ["brew", "services", "restart", "ollama"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    time.sleep(RESTART_SETTLE_S)
    for _ in range(PROBE_ATTEMPTS):
        if probe(model):
            return True
        time.sleep(10)
    return False


def ensure_healthy(model: str = _DEFAULT_MODEL) -> bool:
    """Probe; restart only if genuinely wedged. Cheap on the happy path."""
    return True if probe(model) else restart_and_wait(model)
