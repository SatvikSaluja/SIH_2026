from geocadastra.store.provenance import GENESIS_HASH, append_provenance, verify_chain
from geocadastra.store.schema import Provenance


def test_first_record_chains_from_genesis(db_session):
    row = append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"note": "a"})
    db_session.commit()
    assert row.prev_hash == GENESIS_HASH
    assert len(row.this_hash) == 64


def test_consecutive_records_link_by_hash(db_session):
    r1 = append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"n": 1})
    r2 = append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"n": 2})
    db_session.commit()
    assert r2.prev_hash == r1.this_hash
    assert r1.this_hash != r2.this_hash


def test_verify_chain_passes_on_an_untampered_log(db_session):
    for i in range(5):
        append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="model_inference", payload={"i": i})
    db_session.commit()
    assert verify_chain(db_session) is True


def test_verify_chain_detects_a_tampered_payload(db_session):
    append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"note": "original"})
    append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"note": "b"})
    db_session.commit()
    assert verify_chain(db_session) is True

    first = db_session.query(Provenance).order_by(Provenance.id).first()
    first.payload = {"note": "TAMPERED"}
    db_session.commit()
    assert verify_chain(db_session) is False


def test_verify_chain_detects_a_deleted_middle_record(db_session):
    rows = [
        append_provenance(db_session, edge_id=1, edge_version=1, evidence_type="manual_edit", payload={"i": i})
        for i in range(3)
    ]
    db_session.commit()
    assert verify_chain(db_session) is True

    db_session.delete(db_session.query(Provenance).filter_by(id=rows[1].id).one())
    db_session.commit()
    assert verify_chain(db_session) is False


def test_verify_chain_is_true_on_an_empty_log(db_session):
    assert verify_chain(db_session) is True


def test_hash_does_not_depend_on_created_at_round_tripping():
    """Regression guard for a deliberate design choice: created_at is a
    server/DB-generated timestamp that can gain/lose precision on
    round-trip, so it must not be part of the hashed content -- otherwise
    verify_chain could spuriously flag an untampered row as tampered."""
    from geocadastra.store.provenance import _row_hash

    h1 = _row_hash("prev", 1, 1, "manual_edit", {"a": 1})
    h2 = _row_hash("prev", 1, 1, "manual_edit", {"a": 1})
    assert h1 == h2  # pure function of content, not wall-clock time
