import { useEffect, useState } from "react";
import { api, type WardStatus } from "../api";

interface Props {
  wardJobId: number;
  refreshToken: number;
  onChanged: () => void;
}

export function WardStatusPanel({ wardJobId, refreshToken, onChanged }: Props) {
  const [status, setStatus] = useState<WardStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const load = () =>
    api
      .status(wardJobId)
      .then(setStatus)
      .catch((e) => setError((e as Error).message));

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wardJobId, refreshToken]);

  const runOrResume = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.run(wardJobId);
      setStatus(result);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  if (error) return <div className="panel error">status: {error}</div>;
  if (!status) return <div className="panel">loading status…</div>;

  const counts = status.blocks.reduce<Record<string, number>>((acc, b) => {
    acc[b.status] = (acc[b.status] ?? 0) + 1;
    return acc;
  }, {});
  const failed = status.blocks.filter((b) => b.status === "failed");

  return (
    <div className="panel">
      <h3>
        Ward {status.ward_job_id} — <span className={`badge ${status.status}`}>{status.status}</span>
      </h3>
      <div className="counts">
        {Object.entries(counts).map(([k, v]) => (
          <span key={k} className={`badge ${k}`}>
            {k}: {v}
          </span>
        ))}
      </div>
      <button onClick={runOrResume} disabled={running}>
        {running ? "running…" : "Run / Resume"}
      </button>
      {failed.length > 0 && (
        <div className="failures">
          <h4>Failed blocks</h4>
          <ul className="row-list">
            {failed.map((b) => (
              <li key={b.block_id} className="row error">
                block {b.block_id}: {b.error}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
