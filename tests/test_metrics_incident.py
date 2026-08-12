"""Tests for demo 2's metrics.

The reproposal check is the one that matters. Demo 1's `pick_is_feasible` bug —
a substring metric that could not tell a recommendation from a mention — is
exactly the failure mode available here, so the distinction between recalling an
elimination and re-proposing it is pinned down in both directions.
"""

from demo.metrics_incident import incident_carry, numeric_grounding, reproposes_ruled_out


class TestReproposesRuledOut:
    def test_recalling_an_elimination_is_not_a_reproposal(self):
        answer = (
            "We already ruled out connection pool exhaustion — utilisation "
            "peaked at 34% with zero waits. DNS was also ruled out."
        )
        assert reproposes_ruled_out(answer.lower())["any"] is False

    def test_proposing_a_ruled_out_cause_is_caught(self):
        answer = "Next steps: 1. Check the connection pool for exhaustion. 2. Restart the service."
        result = reproposes_ruled_out(answer.lower())
        assert result["any"] is True
        assert "connection pool" in result["reproposed"]

    def test_each_ruled_out_cause_is_reported_separately(self):
        answer = "Let's investigate DNS resolution and also look at the connection pool again."
        assert set(reproposes_ruled_out(answer.lower())["reproposed"]) == {"connection pool", "dns"}

    def test_one_clean_mention_does_not_excuse_a_separate_proposal(self):
        """A later bare proposal must still count even after an earlier
        properly-marked mention, or an answer could launder a re-proposal by
        mentioning the elimination once at the top."""
        answer = (
            "DNS was ruled out early on. " + ("Filler sentence about the app tier. " * 12)
        ) + "Next, let us re-check DNS resolution end to end."
        assert reproposes_ruled_out(answer.lower())["any"] is True

    def test_no_mention_at_all_is_not_a_reproposal(self):
        answer = "Disable the checkout.fraud_scoring_v2 feature flag."
        assert reproposes_ruled_out(answer.lower())["any"] is False


class TestNumericGrounding:
    def test_corpus_numbers_are_supported(self):
        assert numeric_grounding("The fraud service p99 is 3.9s and there are 64 worker threads.")[
            "grounded"
        ]

    def test_fabricated_magnitude_is_caught(self):
        """The demo-1 failure the blinded judge missed."""
        result = numeric_grounding("Binary quantization reduces memory usage by up to 87%.")
        assert result["grounded"] is False
        assert "87" in result["unsupported"]

    def test_list_numbering_is_not_a_claim(self):
        assert numeric_grounding("1. Disable the flag. 2. Watch p99. 3. Re-enable.")["grounded"]

    def test_times_are_always_checked(self):
        """A time is never list numbering, so a wrong one must not be exempted."""
        assert numeric_grounding("The deploy went out at 09:47 UTC.")["grounded"] is False

    def test_conversation_supplied_numbers_count_as_sources(self):
        answer = "Latency went from 180ms to 4.2s."
        assert numeric_grounding(answer, ["p99 went from 180ms to 4.2s"])["grounded"]


class TestIncidentCarry:
    def _good(self):
        return incident_carry(
            next_steps=(
                "Disable the checkout.fraud_scoring_v2 feature flag — the synchronous "
                "fraud-scoring call added in 2.14 at 13:20 UTC is the cause. We already "
                "ruled out connection pool exhaustion and DNS."
            ),
            buried="The network investigation measured TLS handshake p99 at 41ms.",
            breadth="We investigated the database, the network, and the application tier.",
            conversation_texts=[],
        )

    def test_a_good_answer_scores_across_the_board(self):
        c = self._good()
        assert c["avoids_reproposing_ruled_out"]
        assert c["names_surviving_cause"]
        assert c["proposes_reversible_fix"]
        assert c["uses_corrected_deploy_time"]
        assert c["avoids_stale_deploy_time"]
        assert c["buried_metric_recalled"]
        assert c["breadth_complete"] and c["breadth_subsystems_named"] == 3

    def test_stale_timestamp_without_the_correction_is_penalised(self):
        c = incident_carry(
            next_steps="The deploy at 13:50 UTC introduced the regression.",
            buried="",
            breadth="",
        )
        assert c["uses_corrected_deploy_time"] is False
        assert c["avoids_stale_deploy_time"] is False

    def test_mentioning_both_timestamps_is_not_penalised(self):
        """Citing the canary promotion alongside the corrected deploy time is
        accurate, not a stale-value error."""
        c = incident_carry(
            next_steps="Deployed 13:20 UTC; promoted to 100% at 13:50 UTC.",
            buried="",
            breadth="",
        )
        assert c["uses_corrected_deploy_time"] and c["avoids_stale_deploy_time"]

    def test_partial_breadth_is_counted_not_just_failed(self):
        c = incident_carry(
            next_steps="", buried="", breadth="We looked at the database and the network."
        )
        assert c["breadth_subsystems_named"] == 2
        assert c["breadth_complete"] is False

    def test_reproposed_causes_are_recorded_for_inspection(self):
        c = incident_carry(next_steps="Next: check the connection pool.", buried="", breadth="")
        assert c["_reproposed"] == ["connection pool"]
