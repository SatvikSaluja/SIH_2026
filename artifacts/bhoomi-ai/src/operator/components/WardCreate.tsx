import { useState } from "react";
import { api } from "../api";

interface Props {
  onCreated: (wardJobId: number) => void;
}

// The only ingest wired to the API today is synthetic (/wards/ingest takes
// a seed, not a file) -- jobs/ingest.py can already take real GeoTIFFs +
// recorded parcels, but nothing exposes it over HTTP yet. This form does
// not pretend otherwise.
export function WardCreate({ onCreated }: Props) {
  const [seed, setSeed] = useState(1);
  const [width, setWidth] = useState(120);
  const [height, setHeight] = useState(90);
  const [status, setStatus] = useState<string | null>(null);

  const submit = async () => {
    setStatus("ingesting…");
    try {
      const result = await api.ingest({ seed, width, height });
      setStatus(`ward ${result.ward_job_id}: ${result.n_blocks} blocks, ${result.n_parcels} parcels`);
      onCreated(result.ward_job_id);
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  return (
    <div className="panel">
      <h3>Create a ward</h3>
      <p className="muted">
        Synthetic ingest only, for now (the real GeoTIFF/parcel path exists in the backend but has
        no HTTP route yet). Every downstream stage — capacity refinement, fusion, storage, tiles —
        runs exactly as it would on real data.
      </p>
      <div className="form-grid">
        <label>
          seed
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
        <label>
          width (m)
          <input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} />
        </label>
        <label>
          height (m)
          <input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} />
        </label>
      </div>
      <button onClick={submit}>Ingest</button>
      {status && <div className="status-line">{status}</div>}
    </div>
  );
}
