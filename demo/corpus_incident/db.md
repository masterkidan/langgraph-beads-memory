# Database Investigation — INC-4471

Scope: Postgres primary + 2 replicas behind the checkout service.

## Connection pool
Pool max is 200 connections. Peak utilisation during the incident window
was 34%, with **zero** connection wait events recorded. Pool checkout time
p99 stayed at 0.8ms throughout.

**Conclusion: connection pool exhaustion is RULED OUT.** It was the first
theory raised in the channel and it does not survive the data.

## Query performance
No new entries in the slow query log during the window. Checkout's read
path p99 query time was 8ms before, during, and after the spike —
unchanged. The write path p99 was 11ms, also unchanged.

## Replication
Replica lag peaked at 0.4s, well inside our 5s alerting threshold. No
failover occurred.

## Autovacuum
An autovacuum on `checkout_sessions` began at 14:10 UTC. This is AFTER the
symptom onset at 14:05, so it cannot be causal — it is a consequence of the
elevated write volume from request retries, not a cause.

## Assessment
Nothing in the database layer explains the latency spike. The database was
healthy throughout and was responding normally to a workload that had
itself become abnormal.
