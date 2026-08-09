import uuid

from beads_memory.ids import derive_fact_id, random_fork_suffix, short_id

NS = uuid.UUID("00000000-0000-0000-0000-000000000001")


def test_derive_fact_id_is_deterministic():
    a = derive_fact_id(NS, "msg-1", "hello world")
    b = derive_fact_id(NS, "msg-1", "hello world")
    assert isinstance(a, uuid.UUID) and a == b


def test_derive_fact_id_varies_by_inputs():
    base = derive_fact_id(NS, "msg-1", "hello")
    assert base != derive_fact_id(NS, "msg-2", "hello")
    assert base != derive_fact_id(NS, "msg-1", "bye")
    assert base != derive_fact_id(uuid.uuid4(), "msg-1", "hello")


def test_short_id_format():
    fid = derive_fact_id(NS, "msg-1", "hello")
    s = short_id(fid)
    assert s == f"fact-{fid.hex[:8]}"


def test_random_fork_suffix_shape_and_uniqueness():
    a, b = random_fork_suffix(), random_fork_suffix()
    assert a.startswith("sub-") and len(a) == 12  # "sub-" + 8 hex
    assert a != b
