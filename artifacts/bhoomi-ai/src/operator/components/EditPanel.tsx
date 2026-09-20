import { useEffect, useState } from "react";
import { api, type ConflictRow } from "../api";

interface Props {
  wardJobId: number;
  prefill: { blockId: number; nodeId: number | null; x: number; y: number } | null;
  onApplied: () => void;
}

// Moving a node requires an actual graph node id -- there is no endpoint
// that hands one out for an arbitrary map click (/parcels returns a face's
// resolved ring, not its node ids). A conflict row DOES carry a real
// node_id, so picking a conflict is the reliable path; a manual node id is
// offered for someone who already knows one (e.g. from field notes), never
// invented by this panel.
export function EditPanel({ wardJobId, prefill, onApplied }: Props) {
  const [blockId, setBlockId] = useState(0);
  const [nodeId, setNodeId] = useState<number | "">("");
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const [parcelId, setParcelId] = useState<number | "">("");
  const [asFieldVerification, setAsFieldVerification] = useState(false);
  const [author, setAuthor] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!prefill) return;
    setBlockId(prefill.blockId);
    setX(Number(prefill.x.toFixed(3)));
    setY(Number(prefill.y.toFixed(3)));
    if (prefill.nodeId != null) setNodeId(prefill.nodeId);
  }, [prefill]);

  const submit = async () => {
    if (nodeId === "") {
      setStatus("a real node id is required -- pick a conflict, or enter one you already know");
      return;
    }
    setStatus("submitting…");
    try {
      if (asFieldVerification) {
        if (parcelId === "") {
          setStatus("field verification needs a parcel id");
          return;
        }
        await api.fieldVerification(wardJobId, {
          block_id: blockId,
          node_id: nodeId,
          parcel_id: parcelId,
          x,
          y,
          author: author || undefined,
        });
      } else {
        await api.edit(wardJobId, { block_id: blockId, node_id: nodeId, x, y, author: author || undefined });
      }
      setStatus("applied");
      onApplied();
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  return (
    <div className="panel">
      <h3>Move a node</h3>
      <p className="muted">
        {asFieldVerification
          ? "Applies as a changeset move AND records a durable survey point (source: field_verification)."
          : "Applies as a validated changeset move. Refused if it worsens recorded-area error or block coverage."}
      </p>
      <div className="form-grid">
        <label>
          block id
          <input type="number" value={blockId} onChange={(e) => setBlockId(Number(e.target.value))} />
        </label>
        <label>
          node id (required, must be real)
          <input
            type="number"
            value={nodeId}
            onChange={(e) => setNodeId(e.target.value === "" ? "" : Number(e.target.value))}
          />
        </label>
        <label>
          x (m)
          <input type="number" step="0.001" value={x} onChange={(e) => setX(Number(e.target.value))} />
        </label>
        <label>
          y (m)
          <input type="number" step="0.001" value={y} onChange={(e) => setY(Number(e.target.value))} />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={asFieldVerification}
            onChange={(e) => setAsFieldVerification(e.target.checked)}
          />
          field verification (surveyor confirmed this corner)
        </label>
        {asFieldVerification && (
          <label>
            parcel id
            <input
              type="number"
              value={parcelId}
              onChange={(e) => setParcelId(e.target.value === "" ? "" : Number(e.target.value))}
            />
          </label>
        )}
        <label>
          author (optional)
          <input value={author} onChange={(e) => setAuthor(e.target.value)} />
        </label>
      </div>
      <button onClick={submit}>Apply</button>
      {status && <div className="status-line">{status}</div>}
    </div>
  );
}

export function conflictToPrefill(c: ConflictRow) {
  // -1 is the backend's sentinel for "this conflict has no single node"
  // (area/coverage-level issues) -- never forward it as if it were a real,
  // submittable node id.
  const [x, y] = firstCoordinate(c.geometry) ?? [0, 0];
  return { blockId: c.block_id, nodeId: c.node_id === -1 ? null : c.node_id, x, y };
}

function firstCoordinate(geom: GeoJSON.Geometry): [number, number] | null {
  const g = geom as { type: string; coordinates: unknown };
  let c: unknown = g.coordinates;
  while (Array.isArray(c) && Array.isArray(c[0])) c = c[0];
  return Array.isArray(c) && typeof c[0] === "number" ? (c as [number, number]) : null;
}
