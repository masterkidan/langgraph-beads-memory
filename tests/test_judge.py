from demo.judge import blind_pair


def test_blind_pair_strips_and_randomizes(monkeypatch):
    a = {"condition": "baseline", "final": "answer A"}
    b = {"condition": "treatment", "final": "answer B"}
    monkeypatch.setattr("random.random", lambda: 0.9)  # force swap branch
    pair, mapping = blind_pair(a, b)
    assert set(pair.keys()) == {"X", "Y"}
    assert "condition" not in str(pair)
    assert mapping in (
        {"X": "baseline", "Y": "treatment"},
        {"X": "treatment", "Y": "baseline"},
    )


def test_blind_pair_other_branch_flips_mapping(monkeypatch):
    a = {"condition": "baseline", "final": "answer A"}
    b = {"condition": "treatment", "final": "answer B"}
    monkeypatch.setattr("random.random", lambda: 0.1)  # force no-swap branch
    pair, mapping = blind_pair(a, b)
    assert set(pair.keys()) == {"X", "Y"}
    assert "condition" not in str(pair)
    # Prove the two branches produce opposite X/Y assignments.
    monkeypatch.setattr("random.random", lambda: 0.9)
    _, mapping_swapped = blind_pair(a, b)
    assert mapping != mapping_swapped
