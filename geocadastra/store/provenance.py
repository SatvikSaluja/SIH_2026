"""Append-only, hash-chained evidence log (Stage 2).

Each row's hash covers its own content plus the previous row's hash, so the
log is corruption-evident: recomputing the chain from genesis and comparing
to what's stored (`verify_chain`) catches any accidental edit, deletion, or
reordering. Be precise about what this is NOT: the hash is plain SHA-256,
not a keyed MAC, so anyone with the DB write access this system already
requires to append a row can also recompute a self-consistent chain after
deliberately tampering with one (there's no secret only a legitimate writer
holds). This detects corruption and concurrent-write bugs, not a malicious
insider with write access -- that needs a keyed signature scheme with a key
this application doesn't hold, out of scope here.

Deliberately excludes `created_at` from the hashed content: a timestamp
round-tripped through Postgres can gain/lose precision in ways that would
make `verify_chain` spuriously report tampering that never happened. The
chain guarantees the *evidence content* and *append order* are intact,
which is what this log is actually for.

Appends take a transaction-scoped advisory lock so concurrent legitimate
appends serialize instead of both reading the same tip and forking the
chain (found by review: two real concurrent appends both succeeded with no
error, forking the chain -- verify_chain() then permanently reports "tampered"
with no way to tell that apart from real tampering, on an append-only log).
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from geocadastra.store.schema import Provenance

GENESIS_HASH = "0" * 64
_CHAIN_LOCK_KEY = 918_273_645  # arbitrary fixed key scoping the advisory lock to this one chain


def _row_hash(prev_hash: str, edge_id: int, edge_version: int, evidence_type: str, payload: dict) -> str:
    canonical = json.dumps(
        {
            "prev_hash": prev_hash,
            "edge_id": edge_id,
            "edge_version": edge_version,
            "evidence_type": evidence_type,
            "payload": payload,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_provenance(session: Session, edge_id: int, edge_version: int, evidence_type: str, payload: dict) -> Provenance:
    """Append one evidence record to the end of the global hash chain."""
    # transaction-scoped: released automatically on this transaction's
    # commit or rollback, never needs an explicit unlock
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CHAIN_LOCK_KEY})

    last = session.execute(select(Provenance).order_by(Provenance.id.desc()).limit(1)).scalar_one_or_none()
    prev_hash = last.this_hash if last is not None else GENESIS_HASH
    this_hash = _row_hash(prev_hash, edge_id, edge_version, evidence_type, payload)
    row = Provenance(
        edge_id=edge_id,
        edge_version=edge_version,
        evidence_type=evidence_type,
        payload=payload,
        prev_hash=prev_hash,
        this_hash=this_hash,
    )
    session.add(row)
    session.flush()
    return row


def verify_chain(session: Session) -> bool:
    """Recompute the whole chain from genesis; True iff every row's stored
    hash matches its own content and correctly links to the previous row."""
    rows = session.execute(select(Provenance).order_by(Provenance.id)).scalars().all()
    prev_hash = GENESIS_HASH
    for row in rows:
        if row.prev_hash != prev_hash:
            return False
        if _row_hash(prev_hash, row.edge_id, row.edge_version, row.evidence_type, row.payload) != row.this_hash:
            return False
        prev_hash = row.this_hash
    return True
