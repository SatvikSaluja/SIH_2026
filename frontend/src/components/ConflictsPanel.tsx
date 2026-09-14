import { useEffect, useState } from "react";
import { api, type ConflictRow } from "../api";

interface Props {
  wardJobId: number;
  refreshToken: number;
  onPickConflict: (c: ConflictRow) => void;
}

export function ConflictsPanel({ wardJobId, refreshToken, onPickConflict }: Props) {
  const [conflicts, setConflicts] = useState<ConflictRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .conflicts(wardJobId)
      .then((r) => !cancelled && setConflicts(r.conflicts))
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => {
      cancelled = true;
    };
  }, [wardJobId, refreshToken]);

  if (error) return <div className="panel error">conflicts: {error}</div>;
  if (!conflicts) return <div className="panel">loading conflicts…</div>;

  return (
    <div className="panel">
      <h3>Conflicts ({conflicts.length})</h3>
      {conflicts.length === 0 && <p className="muted">none reported</p>}
      <ul className="row-list">
        {conflicts.map((c) => {
          // node_id -1 is a real sentinel the backend uses for a block-level
          // issue (e.g. area_refinement_unresolved, block_coverage_mismatch)
          // -- there is no single node to move, so this must not be offered
          // as one. Only a conflict with an actual node id routes to Edit.
          const editable = c.node_id !== -1;
          return (
            <li
              key={c.id}
              className={editable ? "row" : "row not-editable"}
              onClick={() => editable && onPickConflict(c)}
              title={editable ? "click to prefill the edit form" : "block-level issue, not a single node"}
            >
              <span className="tag">{c.kind}</span>
              <span>block {c.block_id} · {editable ? `node ${c.node_id}` : "block-level"}</span>
              <span className="muted">{c.sources.join(", ")}</span>
              {c.disagreement_m != null && (
                <span className="muted">{c.disagreement_m.toFixed(3)} m</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
