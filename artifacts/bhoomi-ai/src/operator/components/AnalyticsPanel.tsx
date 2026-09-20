import { useEffect, useState } from "react";
import { api, type AnalyticsResponse } from "../api";

interface Props {
  wardJobId: number;
  refreshToken: number;
}

// One row per settlement style, never pooled into a single mean -- the
// backend's own doc rule ("a mean number across formal and informal blocks
// hides the only failure mode that matters"), carried into the display.
export function AnalyticsPanel({ wardJobId, refreshToken }: Props) {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .analytics(wardJobId)
      .then((r) => !cancelled && setData(r))
      .catch((e) => !cancelled && setError((e as Error).message));
    return () => {
      cancelled = true;
    };
  }, [wardJobId, refreshToken]);

  if (error) return <div className="panel error">analytics: {error}</div>;
  if (!data) return <div className="panel">loading analytics…</div>;

  const styles = Object.keys(data).sort();
  if (styles.length === 0) return <div className="panel">no recorded parcels yet</div>;

  return (
    <div className="panel">
      <h3>Analytics (per settlement style — never pooled)</h3>
      <table className="table">
        <thead>
          <tr>
            <th>style</th>
            <th>boundary error P50 / P90 (m)</th>
            <th>held-out GT pts</th>
            <th>topology validity</th>
            <th>parcel count over/under-seg</th>
            <th>area error (relative, P50)</th>
            <th>area within tolerance</th>
          </tr>
        </thead>
        <tbody>
          {styles.map((style) => {
            const s = data[style];
            return (
              <tr key={style}>
                <td>{style}</td>
                <td>
                  {fmt(s.boundary_position_error_p50)} / {fmt(s.boundary_position_error_p90)}
                </td>
                <td>{s.n_gt_points}</td>
                <td>{s.topology_validity_rate != null ? `${(s.topology_validity_rate * 100).toFixed(1)}%` : "—"}</td>
                <td>
                  +{s.parcel_count.over_segmentation} / -{s.parcel_count.under_segmentation}
                  <span className="muted"> ({s.parcel_count.predicted}/{s.parcel_count.recorded})</span>
                </td>
                <td>{fmt(s.area_error_relative.p50, 3)}</td>
                <td>
                  {s.area_constraints.within_tolerance}/{s.area_constraints.total}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v: number | null, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}
