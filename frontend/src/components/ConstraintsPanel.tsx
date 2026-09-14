import { useEffect, useState } from "react";
import { api, type ConstraintsResponse } from "../api";

interface Props {
  wardJobId: number;
  refreshToken: number;
  onLoaded: (data: ConstraintsResponse) => void;
}

export function ConstraintsPanel({ wardJobId, refreshToken, onLoaded }: Props) {
  const [data, setData] = useState<ConstraintsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .constraints(wardJobId)
      .then((r) => {
        if (cancelled) return;
        setData(r);
        onLoaded(r);
      })
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wardJobId, refreshToken]);

  if (error) return <div className="panel error">constraints: {error}</div>;
  if (!data) return <div className="panel">loading constraints…</div>;

  return (
    <div className="panel">
      <h3>Constraints</h3>
      <div className={`badge ${data.boundary_certification === "not_calibrated" ? "warn" : "ok"}`}>
        boundary certification: {data.boundary_certification}
      </div>
      <div className={`badge ${data.recorded_area_constraints_satisfied ? "ok" : "warn"}`}>
        recorded-area constraints: {data.recorded_area_constraints_satisfied ? "satisfied" : "not satisfied"}
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>block</th>
            <th>parcels</th>
            <th>within tolerance</th>
            <th>unassigned faces</th>
            <th>coverage error (m²)</th>
            <th>ok</th>
          </tr>
        </thead>
        <tbody>
          {data.blocks.map((b) => (
            <tr key={b.block_id}>
              <td>{b.block_id}</td>
              <td>{b.parcels.length}</td>
              <td>
                {b.parcels.filter((p) => p.within_tolerance).length}/{b.parcels.length}
              </td>
              <td>{b.unassigned_face_ids.length}</td>
              <td>{b.block_coverage_error_m2 != null ? b.block_coverage_error_m2.toFixed(4) : "—"}</td>
              <td>{b.constraints_satisfied ? "✓" : "✗"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
