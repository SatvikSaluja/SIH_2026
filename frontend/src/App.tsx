import { useMemo, useState } from "react";
import "./App.css";
import { ParcelMap } from "./components/ParcelMap";
import { ConflictsPanel } from "./components/ConflictsPanel";
import { ConstraintsPanel } from "./components/ConstraintsPanel";
import { AnalyticsPanel } from "./components/AnalyticsPanel";
import { EditPanel, conflictToPrefill } from "./components/EditPanel";
import { CoRegisterPanel } from "./components/CoRegisterPanel";
import { WardCreate } from "./components/WardCreate";
import { WardStatusPanel } from "./components/WardStatusPanel";
import type { ConflictRow, ConstraintsResponse } from "./api";

type Tab = "map" | "conflicts" | "constraints" | "analytics" | "edit" | "coregister";

function App() {
  const [wardJobId, setWardJobId] = useState<number | null>(null);
  const [wardInput, setWardInput] = useState("");
  const [tab, setTab] = useState<Tab>("map");
  const [refreshToken, setRefreshToken] = useState(0);
  const [prefill, setPrefill] = useState<{ blockId: number; nodeId: number | null; x: number; y: number } | null>(
    null
  );
  const [constraints, setConstraints] = useState<ConstraintsResponse | null>(null);

  const toleranceByParcel = useMemo(() => {
    const map = new Map<number, boolean>();
    if (!constraints) return map;
    for (const block of constraints.blocks) {
      for (const p of block.parcels) map.set(p.parcel_id, p.within_tolerance);
    }
    return map;
  }, [constraints]);

  const bump = () => setRefreshToken((t) => t + 1);

  const openWard = () => {
    const id = Number(wardInput);
    if (Number.isFinite(id) && id > 0) setWardJobId(id);
  };

  if (wardJobId == null) {
    return (
      <div className="app">
        <header>
          <h1>GeoCadastra</h1>
          <p className="muted">Parcel-graph reconstruction — operator console</p>
        </header>
        <WardCreate onCreated={setWardJobId} />
        <div className="panel">
          <h3>Open an existing ward</h3>
          <div className="form-grid">
            <label>
              ward job id
              <input value={wardInput} onChange={(e) => setWardInput(e.target.value)} />
            </label>
          </div>
          <button onClick={openWard}>Open</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header>
        <h1>GeoCadastra</h1>
        <button className="link" onClick={() => setWardJobId(null)}>
          ← switch ward
        </button>
      </header>

      <WardStatusPanel wardJobId={wardJobId} refreshToken={refreshToken} onChanged={bump} />

      <nav className="tabs">
        {(["map", "conflicts", "constraints", "analytics", "edit", "coregister"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>

      {tab === "map" && (
        <ParcelMap
          wardJobId={wardJobId}
          toleranceByParcel={toleranceByParcel}
          refreshToken={refreshToken}
          onMapClick={(blockId, x, y) => {
            setPrefill({ blockId, nodeId: null, x, y });
            setTab("edit");
          }}
        />
      )}
      {tab === "conflicts" && (
        <ConflictsPanel
          wardJobId={wardJobId}
          refreshToken={refreshToken}
          onPickConflict={(c: ConflictRow) => {
            setPrefill(conflictToPrefill(c));
            setTab("edit");
          }}
        />
      )}
      {tab === "constraints" && (
        <ConstraintsPanel wardJobId={wardJobId} refreshToken={refreshToken} onLoaded={setConstraints} />
      )}
      {tab === "analytics" && <AnalyticsPanel wardJobId={wardJobId} refreshToken={refreshToken} />}
      {tab === "edit" && <EditPanel wardJobId={wardJobId} prefill={prefill} onApplied={bump} />}
      {tab === "coregister" && <CoRegisterPanel wardJobId={wardJobId} onApplied={bump} />}
    </div>
  );
}

export default App;
