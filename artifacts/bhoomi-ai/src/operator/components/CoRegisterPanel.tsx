import { useState } from "react";
import { api } from "../api";

interface Props {
  wardJobId: number;
  onApplied: () => void;
}

// The backend never rejects a bad-but-well-posed alignment -- a large
// residual just widens fusion's trust in the legacy layer going forward.
// It DOES reject degenerate points (all identical, collinear), which is a
// client mistake, not an honest "the alignment isn't great" answer.
export function CoRegisterPanel({ wardJobId, onApplied }: Props) {
  const [rows, setRows] = useState([{ sx: "", sy: "", dx: "", dy: "" }]);
  const [status, setStatus] = useState<string | null>(null);

  const addRow = () => setRows([...rows, { sx: "", sy: "", dx: "", dy: "" }]);
  const setCell = (i: number, key: keyof (typeof rows)[number], value: string) =>
    setRows(rows.map((r, idx) => (idx === i ? { ...r, [key]: value } : r)));

  const submit = async () => {
    const points = rows.map((r) => [Number(r.sx), Number(r.sy), Number(r.dx), Number(r.dy)] as [number, number, number, number]);
    if (points.length < 3) {
      setStatus("need at least 3 control points");
      return;
    }
    setStatus("fitting…");
    try {
      const result = await api.coregister(wardJobId, { control_points: points });
      setStatus(`fit applied, RMS residual ${result.residual_m.toFixed(4)} m`);
      onApplied();
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  return (
    <div className="panel">
      <h3>Co-registration</h3>
      <p className="muted">
        Fits a best-fit affine from legacy-layer points to this ward's reference frame. Never rejects
        a poor fit outright -- widens legacy evidence's uncertainty instead. Requires ≥3 non-degenerate points.
      </p>
      <table className="table">
        <thead>
          <tr>
            <th>src x</th>
            <th>src y</th>
            <th>dst x</th>
            <th>dst y</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {(["sx", "sy", "dx", "dy"] as const).map((key) => (
                <td key={key}>
                  <input
                    type="number"
                    step="0.001"
                    value={r[key]}
                    onChange={(e) => setCell(i, key, e.target.value)}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={addRow}>+ row</button>
      <button onClick={submit}>Fit &amp; apply</button>
      {status && <div className="status-line">{status}</div>}
    </div>
  );
}
