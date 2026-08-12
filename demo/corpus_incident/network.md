# Network Investigation — INC-4471

Scope: edge load balancer, service mesh, DNS, inter-AZ paths.

## Load balancer
The LB returned no 5xx of its own. Every error it logged was an origin
error passed through from the checkout service. Upstream connection reuse
was normal.

## Packet loss and retransmits
Packet loss measured 0.00% across all monitored paths. TCP retransmit rate
was 0.02%, identical to the trailing 7-day baseline.

## DNS
DNS resolution p99 measured 3.1ms during the window, unchanged from
baseline. No resolver timeouts, no NXDOMAIN spikes.

**Conclusion: DNS is RULED OUT as a contributing factor.**

## TLS
TLS handshake p99 was 41ms during the window. This is in line with the
trailing baseline and was not investigated further.

## Inter-AZ
Cross-AZ round-trip latency held at 1.2ms. No AZ was degraded.

## Assessment
The network layer was healthy in every dimension measured. Latency
observed at the LB is the checkout service's own response time, not
transport delay.
