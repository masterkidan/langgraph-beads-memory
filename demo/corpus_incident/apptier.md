# Application Tier Investigation — INC-4471

Scope: checkout service, release 2.14.

## What release 2.14 changed
Release 2.14 added a **synchronous call to the fraud-scoring service**
inside the checkout request path, immediately before order commit. It is
guarded by the feature flag `checkout.fraud_scoring_v2`.

The call was configured with a 5000ms timeout, **no circuit breaker, and no
fallback path**. On timeout the request fails rather than proceeding
without a score.

## Fraud-scoring service behaviour
Under production checkout volume the fraud-scoring service p99 is 3.9s — it
was sized for the batch workload it previously served, not for a
synchronous per-request call. Its own dashboards show it behaving to spec;
it is not itself broken.

## Thread pool
The checkout service runs 64 worker threads. With each checkout blocking up
to 5s on fraud scoring, all 64 were observed blocked simultaneously.
Requests then queued at the acceptor, which is what turned a 3.9s
dependency into a 4.2s user-visible p99 and produced the 7% error rate as
queued requests aged out.

## Ruled out within this tier
Heap usage was stable at 61%. GC pause p99 was 12ms, unchanged. No memory
leak signature.

## Reversibility
`checkout.fraud_scoring_v2` can be disabled at runtime through the flag
service. It takes effect in seconds and requires no deploy and no restart.
A full rollback of 2.14 is also possible but takes roughly 25 minutes.

## Assessment
This is the cause. The synchronous fraud-scoring call introduced in 2.14
put a 3.9s p99 dependency in the checkout hot path with no circuit breaker.
