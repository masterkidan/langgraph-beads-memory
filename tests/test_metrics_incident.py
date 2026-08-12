"""Tests for demo 2's metrics.

The reproposal check is the one that matters. Demo 1's `pick_is_feasible` bug —
a substring metric that could not tell a recommendation from a mention — is
exactly the failure mode available here, so the distinction between recalling an
elimination and re-proposing it is pinned down in both directions.
"""

import pytest

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
                "fraud-scoring call added in 2.14 is the cause. We already "
                "ruled out connection pool exhaustion and DNS."
            ),
            buried="The network investigation measured TLS handshake p99 at 41ms.",
            breadth="We investigated the database, the network, and the application tier.",
            timeline="Release 2.14 went out at 13:20 UTC.",
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
            next_steps="",
            buried="",
            breadth="",
            timeline="The deploy went out at 13:50 UTC.",
        )
        assert c["uses_corrected_deploy_time"] is False
        assert c["avoids_stale_deploy_time"] is False

    def test_mentioning_both_timestamps_is_not_penalised(self):
        """Citing the canary promotion alongside the corrected deploy time is
        accurate, not a stale-value error."""
        c = incident_carry(
            next_steps="",
            buried="",
            breadth="",
            timeline="Deployed 13:20 UTC; promoted to 100% at 13:50 UTC.",
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


class TestReproposalOnRealProse:
    """Sentences taken verbatim from real runs.

    Every one of these was, at some point, scored wrongly. They are pinned here
    because the metric decides demo 2's headline result, and a false positive
    on it manufactures a separation that does not exist -- which is exactly
    what happened in an N=1 calibration run before these landed.
    """

    REAL_RECALLS = [
        # Scored as a re-proposal because "within acceptable thresholds" was not
        # in the elimination vocabulary. It is a correct recall.
        "Connection pool exhaustion, query performance, replication lag, and "
        "autovacuum activity were all within acceptable thresholds.",
        # Scored as a re-proposal because the window contained "the CHECKOUT
        # service's own response time" and "check" was matched as a substring --
        # the same prefix-matching bug as classify_fragment's "Checkout"/"check".
        "The latency is due to the checkout service's own response time, not "
        "transport delay. No database layer issues were identified. Connection "
        "pool exhaustion was within acceptable thresholds.",
    ]

    @pytest.mark.parametrize("text", REAL_RECALLS)
    def test_correct_recall_is_not_flagged(self, text):
        assert reproposes_ruled_out(text.lower())["any"] is False

    @pytest.mark.parametrize(
        "text",
        [
            "Next steps: 1. Check the connection pool for exhaustion.",
            "I recommend we investigate DNS resolution end to end.",
            "We should re-examine the connection pool.",
        ],
    )
    def test_genuine_reproposals_are_still_caught(self, text):
        assert reproposes_ruled_out(text.lower())["any"] is True

    def test_checkout_does_not_read_as_the_verb_check(self):
        """Word boundaries, not substrings -- learned twice on this project."""
        text = "The checkout service is slow; connection pool exhaustion was eliminated."
        assert reproposes_ruled_out(text)["any"] is False


class TestTimelineIsScoredInIsolation:
    """The corrected timestamp is scored from its own question.

    Folding it into the conv-3 next-steps question was measured to crowd that
    answer out: both arms stopped naming the surviving cause and stopped
    proposing the reversible fix.
    """

    def test_a_timestamp_in_the_next_steps_answer_does_not_count(self):
        c = incident_carry(
            next_steps="The deploy at 13:20 UTC is implicated.",
            buried="",
            breadth="",
            timeline="",
        )
        assert c["uses_corrected_deploy_time"] is False

    def test_it_counts_when_answered_in_its_own_turn(self):
        c = incident_carry(
            next_steps="", buried="", breadth="", timeline="Release 2.14 went out at 13:20 UTC."
        )
        assert c["uses_corrected_deploy_time"] is True
