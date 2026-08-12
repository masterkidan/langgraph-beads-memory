"""Demo 2: a production incident investigated across four threads.

Chosen to test whether the demo-1 result generalises to a domain that was NOT
designed around a planted correction. Here the mechanism gets exercised the way
a real investigation exercises it: hypotheses are eliminated, and an agent that
forgets an elimination wastes an on-call engineer's time re-testing it.

The question set is deliberately mixed. Two questions favour a typed fact graph
(don't re-propose a ruled-out cause; carry a corrected timestamp) and two favour
a flat blob store (recall an incidental measurement; recall investigation
breadth). Predictions for all four are pre-registered in
`results/2026-08-11-demo2-preregistration.md`, written before the first run.
"""

# Same delegation discipline as demo 1, and for the same measured reason: without
# it the agent delegates on a turn that only reports symptoms, which both wastes
# most of the run and corrupts the delegation measurement. Identical wording goes
# to both conditions.
INCIDENT_SYSTEM_PROMPT = (
    "You are the incident commander for a production incident. You have memory "
    "tools: when you reach a conclusion — especially when a hypothesis is ruled "
    "out — record it with remember_fact. When a fact you were given earlier "
    "turns out to be wrong, record the corrected value with remember_fact using "
    "relation='supersedes' and the old fact's short id from your Memory "
    "context.\n\n"
    "Delegation rules — follow exactly:\n"
    "- A SITUATION REPORT IS NOT A REQUEST TO INVESTIGATE. When the user "
    "describes symptoms, timings, error rates, deploys, or constraints, your "
    "entire job for that turn is: acknowledge, record each fact with "
    "remember_fact, and STOP. Do not delegate. Do not call read_document. Wait "
    "to be asked.\n"
    "- Only delegate to your investigator sub-agents on a turn where the user "
    "explicitly asks you to investigate, dig in, or delegate.\n"
    "- If the user is correcting a detail, or asking a question you can answer "
    "from your Memory context, do NOT delegate and do NOT call read_document.\n"
    "- Never delegate the same subsystem twice; if an investigator already "
    "reported on it, use that conclusion from your Memory context instead."
)

INVESTIGATOR_SYSTEM_PROMPT = (
    "You are a focused subsystem investigator on an incident. Read the relevant "
    "documents thoroughly with read_document, record notable findings with "
    "remember_fact — including anything you RULE OUT and why — and you MUST "
    "finish by calling conclude_task with a summary of your conclusion for the "
    "incident commander."
)

# Appended only for the `treatment-subrecall` arm. This is the sub-agent leg of
# the hypothesis, which demo 1 never tested: the tool has always been bound to
# the orchestrator, but no scenario prompt ever mentioned it, so it was
# unreachable in practice.
SUBRECALL_PROMPT_SUFFIX = (
    "\n- When asked about a specific measurement or detail that one of your "
    "investigators looked into, call recall_from_subagents to read what that "
    "investigator actually recorded, rather than answering from the summary "
    "alone."
)

SUBSYSTEMS = ("db", "network", "apptier")

CONVERSATIONS = [
    (
        "conv-1",
        [
            "We have a production incident. Checkout p99 latency went from 180ms "
            "to 4.2s starting at 14:05 UTC, and the error rate went from 0.3% to "
            "7%. We deployed release 2.14 at 13:50 UTC. Two constraints on the "
            "response: whatever we try must be reversible within 30 minutes, and "
            "we are not taking a full outage to fix this.",
        ],
    ),
    (
        "conv-2",
        [
            "Please investigate this properly. Delegate the db, network, and "
            "apptier subsystems to your investigators, then give me your "
            "synthesis.",
            # The correction beat. Natural to an incident — deploy timelines are
            # routinely misreported in the first hour — rather than planted.
            "Correction on the timeline: the deploy actually went out at 13:20 "
            "UTC, not 13:50. The 13:50 timestamp was the canary being promoted "
            "to 100% of traffic.",
        ],
    ),
    (
        "conv-3",
        [
            # Q1 — predicted to favour the treatment. Re-proposing a ruled-out
            # cause is the concrete cost of forgetting, and it is what the typed
            # invalidation is supposed to prevent.
            "New shift taking over. Given everything we've established, what "
            "should we try next, and why? Be specific about what we already "
            "ruled out so I don't repeat work.",
        ],
    ),
    (
        "conv-4",
        [
            # Q2 — predicted to favour the BASELINE. The TLS handshake number is
            # incidental: it is not the reason anything was ruled out, so an
            # investigator summarising its conclusion has little reason to carry
            # it up. A flat blob store retains it for free.
            "One more thing for the postmortem — what exactly did the network "
            "investigation measure for TLS handshake p99?",
            # Q3 — also predicted to favour the baseline. Breadth recall rewards
            # keeping everything over ranking the most relevant few.
            "And list everything we investigated and ruled out, with the "
            "measurement behind each one.",
        ],
    ),
]

# Ground truth for the objective metrics.
PLANTED = {
    # Causes eliminated by the investigations. Naming one as something still to
    # try is the failure this scenario is built to detect.
    "ruled_out_terms": {
        "connection pool": ["connection pool", "connection-pool", "pool exhaustion"],
        "dns": ["dns"],
    },
    # Language that marks a mention as an elimination rather than a proposal.
    "elimination_markers": [
        "ruled out",
        "ruled-out",
        "eliminated",
        "excluded",
        "not the cause",
        "not a cause",
        "not the culprit",
        "was healthy",
        "were healthy",
        "no evidence",
        "already investigated",
        "already checked",
        "already ruled",
        "confirmed healthy",
        "is fine",
        "were fine",
        "was normal",
        "were normal",
        "unchanged",
        "cannot be causal",
        "not contributing",
        "did not contribute",
    ],
    # The cause that survives.
    "surviving_cause_variants": [
        "fraud",
        "fraud-scoring",
        "fraud scoring",
    ],
    "circuit_breaker_variants": ["circuit breaker", "circuit-breaker"],
    # The reversible remediation the constraints demand.
    "reversible_fix_variants": [
        "feature flag",
        "feature-flag",
        "checkout.fraud_scoring_v2",
        "fraud_scoring_v2",
        "flag service",
        "disable the flag",
    ],
    "corrected_deploy_time_variants": ["13:20"],
    "stale_deploy_time_variants": ["13:50"],
    # Incidental measurement, network.md only.
    "buried_metric_variants": [["41ms", "41 ms", "41 milliseconds"]],
    # Breadth: how many of the three subsystems get named.
    "subsystem_variants": {
        "db": ["database", "db ", "postgres"],
        "network": ["network", "dns", "packet loss", "load balancer"],
        "apptier": ["app tier", "apptier", "application tier", "checkout service", "2.14"],
    },
}
