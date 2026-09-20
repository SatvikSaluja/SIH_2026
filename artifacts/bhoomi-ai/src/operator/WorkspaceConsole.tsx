import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
// Stylesheet is the old standalone app's, mechanically scoped under
// .workspace-console -- it styles bare `body`/`button`/`h1` and reuses ten
// class names the shell already owns (.brand, .topbar, .nav-item, ...), so
// unscoped it would restyle every other page in this app.
import "./workspace-console.css";
import OperatorConsole from "./OperatorConsole";
import { request, asset, jobAsset } from "./workspace";
import type {
  Catalog,
  Tile,
  Detail,
  Evidence,
  Candidate,
  Job,
  Review,
  Training,
  Epoch,
} from "./workspace";

type Page =
  | "Overview"
  | "Datasets"
  | "Inference"
  | "Evidence"
  | "Review queue"
  | "Vision assistant"
  | "Training"
  | "Ward operations";
const pages: Page[] = [
  "Overview",
  "Datasets",
  "Inference",
  "Evidence",
  "Review queue",
  "Vision assistant",
  "Training",
  "Ward operations",
];
const symbols = ["◈", "▦", "⌁", "◉", "☷", "✧", "↗", "⌘"];
const descriptions: Record<Page, string> = {
  Overview:
    "A clear view of the data, evidence and decisions behind every parcel.",
  Datasets: "Explore real imagery and the references that make it useful.",
  Inference: "Run saved weights. Inspect the prediction. Measure what matters.",
  Evidence: "Inspect the physical evidence along each reference boundary.",
  "Review queue":
    "Turn uncertain image evidence into a traceable human decision.",
  "Vision assistant":
    "A second look at ambiguous features. You remain the reviewer.",
  Training: "Follow actual experiments, checkpoints and held-out results.",
  "Ward operations":
    "Reconstruct parcels, inspect constraints and apply recorded edits.",
};
const percent = (v?: number) =>
  v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
const pretty = (s: string) => s.replaceAll("_", " ").replaceAll("nz ", "NZ ");

function ImageMap({
  detail,
  source,
  region,
}: {
  detail: Detail;
  source: string;
  region?: Candidate | null;
}) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const overlay = useRef<L.ImageOverlay | null>(null);
  const box = useRef<L.Rectangle | null>(null);
  useEffect(() => {
    if (!container.current) return;
    const m = L.map(container.current, {
      crs: L.CRS.Simple,
      zoomAnimation: false,
      fadeAnimation: false,
      markerZoomAnimation: false,
      minZoom: -4,
      zoomSnap: 0.25,
      maxZoom: 3,
      attributionControl: false,
    });
    map.current = m;
    return () => {
      m.stop();
      m.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    overlay.current?.remove();
    const bounds: L.LatLngBoundsExpression = [
      [0, 0],
      [detail.height, detail.width],
    ];
    overlay.current = L.imageOverlay(source, bounds).addTo(m);
    m.fitBounds(bounds, { padding: [16, 16], animate: false });
    m.setZoom(m.getZoom() + 1, { animate: false });
    const resize = new ResizeObserver(() => {
      if (map.current === m && container.current?.isConnected)
        m.invalidateSize({ animate: false });
    });
    if (container.current) resize.observe(container.current);
    return () => resize.disconnect();
  }, [detail.id, detail.width, detail.height, source]);
  useEffect(() => {
    box.current?.remove();
    if (region && map.current) {
      box.current = L.rectangle(
        [
          [detail.height - region.y - region.height, region.x],
          [detail.height - region.y, region.x + region.width],
        ],
        { color: "#ffc766", weight: 2, fillOpacity: 0.1 },
      ).addTo(map.current);
      map.current.fitBounds(box.current.getBounds(), {
        padding: [100, 100],
        maxZoom: 1,
        animate: false,
      });
    }
  }, [region, detail.height]);
  return (
    <div
      className="image-map"
      ref={container}
      aria-label="Interactive aerial imagery. Use plus and minus controls to zoom."
    />
  );
}
function Sparkline({ history }: { history: Epoch[] }) {
  const values = history
    .map((e) => e.validation?.loss)
    .filter((v) => Number.isFinite(v));
  if (values.length < 2)
    return (
      <p className="muted">At least two recorded epochs needed for a curve.</p>
    );
  const lo = Math.min(...values),
    hi = Math.max(...values);
  const points = values
    .map(
      (v, i) =>
        `${10 + (i / (values.length - 1)) * 780},${145 - ((v - lo) / Math.max(hi - lo, 0.01)) * 120}`,
    )
    .join(" ");
  return (
    <div className="chart">
      <span>Validation loss · within this run only</span>
      <svg
        role="img"
        aria-label={`Validation loss over ${values.length} epochs`}
        viewBox="0 0 800 170"
      >
        <path
          d="M10 25H790 M10 85H790 M10 145H790"
          stroke="#e5eae7"
          fill="none"
        />
        <polyline
          points={points}
          fill="none"
          stroke="#19876b"
          strokeWidth="3"
        />
      </svg>
      <div className="between">
        <small>Epoch 1</small>
        <small>Epoch {values.length}</small>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("Overview");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [dataset, setDataset] = useState("");
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState("");
  const [split, setSplit] = useState("");
  const [tile, setTile] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);
  const [layer, setLayer] = useState("rgb");
  const [checkpoint, setCheckpoint] = useState("");
  const [compare, setCompare] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [region, setRegion] = useState<Candidate | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [note, setNote] = useState("");
  const [advice, setAdvice] = useState("");
  const [training, setTraining] = useState<Training[]>([]);
  const [run, setRun] = useState("");
  const [epochs, setEpochs] = useState(5);
  const [batch, setBatch] = useState(4);
  const [visibilityAware, setVisibilityAware] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const guard = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const refresh = async () => {
    const [c, j, r, t] = await Promise.all([
      request<Catalog>("/catalog"),
      request<Job[]>("/jobs"),
      request<Review[]>("/reviews"),
      request<Training[]>("/training"),
    ]);
    setCatalog(c);
    setJobs(j);
    setReviews(r);
    setTraining(t);
    setDataset(
      (prev) =>
        prev ||
        c.datasets.find((x) => x.id === "nz_multitask_real_v2")?.id ||
        c.datasets[0]?.id ||
        "",
    );
    setCheckpoint(
      (prev) =>
        prev ||
        c.checkpoints.find(
          (x) => x.id === "nz_multitask_real_v2_weighted/best.pt",
        )?.id ||
        c.checkpoints[0]?.id ||
        "",
    );
    setRun((prev) => prev || t.at(-1)?.id || "");
  };
  useEffect(() => {
    void guard(refresh);
    const timer = setInterval(() => {
      void refresh().catch(() => {});
    }, 10000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!dataset) return;
    let active = true;
    setError("");
    request<{ tiles: Tile[]; total: number }>(
      `/tiles?dataset=${encodeURIComponent(dataset)}&search=${encodeURIComponent(search)}&split=${split}&offset=${offset}`,
    )
      .then((x) => {
        if (active) {
          setTiles(x.tiles);
          setTotal(x.total);
          setTile((old) =>
            x.tiles.some((t) => t.id === old) ? old : x.tiles[0]?.id || "",
          );
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [dataset, search, split, offset]);
  useEffect(() => {
    setDetail(null);
    setEvidence(null);
    setRegion(null);
    setLayer("rgb");
    setAdvice("");
    if (!dataset || !tile) return;
    let active = true;
    request<Detail>(
      `/tiles/${encodeURIComponent(dataset)}/${encodeURIComponent(tile)}`,
    )
      .then((d) => {
        if (active) setDetail(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [dataset, tile]);
  useEffect(() => {
    if (
      !detail ||
      !["Evidence", "Review queue", "Vision assistant"].includes(page)
    )
      return;
    let active = true;
    request<Evidence>(`/tiles/${dataset}/${tile}/evidence`)
      .then((x) => {
        if (active) {
          setEvidence(x);
          setRegion(x.candidates[0] || null);
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [detail, page, dataset, tile]);
  const ds = catalog?.datasets.find((d) => d.id === dataset);
  const currentJob = jobs.find((j) => j.id === selectedJob);
  const completed = jobs.filter(
    (j) =>
      j.kind === "inference" &&
      j.status === "completed" &&
      j.dataset === dataset &&
      j.tile === tile,
  );
  const currentRun = training.find((t) => t.id === run);
  const last = currentRun?.history.at(-1);
  const image = detail ? asset(dataset, tile, layer) : "";
  const launch = () =>
    guard(async () => {
      const selected = [
        checkpoint,
        ...(compare && compare !== checkpoint ? [compare] : []),
      ];
      for (const cp of selected) {
        const j = await request<Job>("/inference", {
          dataset,
          tile,
          checkpoint: cp,
          threshold_m: 0.3,
        });
        setSelectedJob(j.id);
      }
      setNotice(
        "Inference queued. Results appear after actual model execution.",
      );
      await refresh();
    });
  const saveReview = (decision: string) =>
    guard(async () => {
      if (!region) return;
      await request("/reviews", {
        dataset,
        tile,
        region: region.id,
        decision,
        note,
      });
      setNote("");
      setNotice("Review saved. Training labels remain unchanged.");
      await refresh();
    });
  const sendVision = () =>
    guard(async () => {
      if (!region) return;
      const result = await request<Review>("/vision", {
        dataset,
        tile,
        x: region.x,
        y: region.y,
      });
      setAdvice(result.advice || "No textual assessment returned.");
      await refresh();
    });
  const selectDataset = (id: string) => {
    setDataset(id);
    setOffset(0);
    setTile("");
    setSelectedJob("");
  };
  return (
    <div className="workspace-console">
    <div className="workspace-shell">
      <aside className="rail">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("Overview");
          }}
        >
          <span className="brand-mark">
            G<span>◈</span>
          </span>
          <span>
            GeoCadastra<small>LAND INTELLIGENCE</small>
          </span>
        </a>
        <div className="workspace-label">
          RESEARCH WORKSPACE <span>01</span>
        </div>
        <nav aria-label="Main navigation">
          {pages.map((p, i) => (
            <button
              key={p}
              className={page === p ? "nav-item selected" : "nav-item"}
              onClick={() => {
                setPage(p);
                setError("");
                setNotice("");
              }}
            >
              <span className="nav-symbol">{symbols[i]}</span>
              {p}
              {page === p && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="rail-bottom">
          <div className="signal">
            <i /> Local workspace
          </div>
          <p>
            Traceable inputs.
            <br />
            Reviewable decisions.
          </p>
          <div className="profile">
            <span>GC</span>
            <div>
              Research operator<small>Local session</small>
            </div>
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <span>
            Workspace <b>/</b> {page}
          </span>
          <div>
            <span className="status-pill neutral">
              Research · not calibrated
            </span>
            <button
              className="icon-button"
              onClick={() => void guard(refresh)}
              aria-label="Refresh workspace"
            >
              ↻
            </button>
          </div>
        </header>
        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                GEOCADASTRA /{" "}
                {page === "Overview" ? "MISSION CONTROL" : page.toUpperCase()}
              </div>
              <h1>
                {page === "Overview"
                  ? "Every boundary, backed by evidence."
                  : page}
              </h1>
              <p>{descriptions[page]}</p>
            </div>
            <button
              className="secondary"
              onClick={() =>
                setPage(page === "Datasets" ? "Inference" : "Datasets")
              }
            >
              {page === "Datasets"
                ? "Open inference studio ↗"
                : "Explore datasets ↗"}
            </button>
          </div>
          {error && (
            <div className="message error" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          {notice && (
            <div className="message success" role="status">
              {notice}
              <button
                onClick={() => setNotice("")}
                aria-label="Dismiss notification"
              >
                ×
              </button>
            </div>
          )}
          {!catalog && !error && (
            <div className="empty">Connecting to the workspace…</div>
          )}
          {page !== "Ward operations" && catalog && (
            <div className="context-bar">
              <span className="context-icon">▦</span>
              <label>
                Dataset
                <select
                  value={dataset}
                  onChange={(e) => selectDataset(e.target.value)}
                >
                  {catalog.datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {pretty(d.id)} · {d.count} tiles
                    </option>
                  ))}
                </select>
              </label>
              <div className="context-divider" />
              <span>{ds?.crs || "CRS not recorded"}</span>
              <span className="status-pill">
                {ds?.status === "in_progress"
                  ? "Collection in progress"
                  : "Dataset snapshot"}
              </span>
              <span className="context-end">
                {ds?.count?.toLocaleString()} source tiles
              </span>
            </div>
          )}
          {page === "Overview" && (
            <>
              <div className="stats-grid">
                <Stat
                  label="SOURCE TILES"
                  value={ds?.count?.toLocaleString() || "—"}
                  detail="In the selected dataset"
                />
                <Stat
                  label="SAVED CHECKPOINTS"
                  value={String(catalog?.checkpoints.length ?? 0)}
                  detail="Available for model inspection"
                />
                <Stat
                  label="COMPLETED INFERENCES"
                  value={String(
                    jobs.filter(
                      (j) => j.kind === "inference" && j.status === "completed",
                    ).length,
                  )}
                  detail="Executed in this workspace"
                />
                <Stat
                  label="HUMAN REVIEWS"
                  value={String(
                    reviews.filter((r) => r.decision !== "advisory_only")
                      .length,
                  )}
                  detail="Persisted review decisions"
                />
              </div>
              <div className="overview-grid">
                <section className="panel map-panel">
                  <div className="section-title">
                    <div>
                      <span className="eyebrow">AREA EXPLORER</span>
                      <h2>{tile || "Select an area"}</h2>
                    </div>
                    <span className="status-pill green">
                      Real aerial imagery
                    </span>
                  </div>
                  {detail ? (
                    <>
                      <div className="map-wrap">
                        <ImageMap detail={detail} source={image} />
                        <div className="map-badge">
                          NZ · {detail.gsd_m} m / pixel
                        </div>
                        <div className="map-caption">
                          {detail.width} × {detail.height} pixels{" "}
                          <span>{detail.split} split</span>
                        </div>
                      </div>
                      <div className="panel-footer">
                        <span>
                          Inspect imagery, reference lines and measured height.
                        </span>
                        <button
                          className="text-button"
                          onClick={() => setPage("Datasets")}
                        >
                          Open explorer →
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="empty">
                      Choose a dataset with available imagery.
                    </div>
                  )}
                </section>
                <div className="overview-side">
                  <section className="panel">
                    <span className="eyebrow">YOUR NEXT STEP</span>
                    <h2>From image to evidence.</h2>
                    <p className="muted">
                      Follow the data through prediction and review.
                    </p>
                    {[
                      ["01", "Inspect source imagery", "Datasets"],
                      ["02", "Run a saved checkpoint", "Inference"],
                      ["03", "Review uncertain sections", "Review queue"],
                    ].map(([n, title, p]) => (
                      <button
                        className="step"
                        key={n}
                        onClick={() => setPage(p as Page)}
                      >
                        <span>{n}</span>
                        <strong>{title}</strong>
                        <b>↗</b>
                      </button>
                    ))}
                  </section>
                  <section className="panel dark-panel">
                    <span className="eyebrow">EVIDENCE, NOT ASSUMPTIONS</span>
                    <h2>A line on a map isn’t always visible on the ground.</h2>
                    <p>
                      Image support and vision advice help reviewers inspect
                      boundaries. Neither certifies ownership or survey
                      accuracy.
                    </p>
                    <button
                      className="text-button"
                      onClick={() => setPage("Evidence")}
                    >
                      Inspect image support →
                    </button>
                  </section>
                </div>
              </div>
              <Jobs
                jobs={jobs.slice(0, 5)}
                onSelect={(j) => {
                  selectDataset(j.dataset);
                  setTile(j.tile || "");
                  setSelectedJob(j.id);
                  setPage("Inference");
                }}
              />
            </>
          )}
          {page === "Datasets" && (
            <>
              <div className="filter-row">
                <input
                  aria-label="Search tiles"
                  placeholder="Search a tile ID…"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setOffset(0);
                  }}
                />
                <select
                  aria-label="Dataset split"
                  value={split}
                  onChange={(e) => {
                    setSplit(e.target.value);
                    setOffset(0);
                  }}
                >
                  <option value="">All splits</option>
                  <option>train</option>
                  <option>val</option>
                  <option>test</option>
                </select>
                <span>{total} matching tiles</span>
              </div>
              <div className="tile-grid">
                {tiles.map((t) => (
                  <button
                    className={`tile-card ${tile === t.id ? "chosen" : ""}`}
                    key={t.id}
                    onClick={() => setTile(t.id)}
                  >
                    <img
                      loading="lazy"
                      src={asset(dataset, t.id, "rgb", true)}
                      alt={`Aerial tile ${t.id}`}
                    />
                    <div>
                      <strong>{t.id}</strong>
                      <span className="status-pill">{t.split}</span>
                    </div>
                    <small>
                      {t.gsd_m} m / pixel ·{" "}
                      {t.parcels === undefined
                        ? "Parcel count not recorded"
                        : `${t.parcels} parcels`}
                    </small>
                  </button>
                ))}
              </div>
              {!tiles.length && (
                <div className="empty">No tiles match these filters.</div>
              )}
              <div className="pagination">
                <button
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - 48))}
                >
                  ← Previous
                </button>
                <span>
                  {offset + 1}–{Math.min(offset + 48, total)} of {total}
                </span>
                <button
                  disabled={offset + 48 >= total}
                  onClick={() => setOffset(offset + 48)}
                >
                  Next →
                </button>
              </div>
              {detail && (
                <section className="panel">
                  <div className="section-title">
                    <h2>Selected: {detail.id}</h2>
                    <button onClick={() => setPage("Inference")}>
                      Inspect & run inference ↗
                    </button>
                  </div>
                  <Details detail={detail} />
                </section>
              )}
            </>
          )}
          {[
            "Inference",
            "Evidence",
            "Review queue",
            "Vision assistant",
          ].includes(page) && (
            <>
              <div className="filter-row">
                <label>
                  Source tile{" "}
                  <select
                    value={tile}
                    onChange={(e) => setTile(e.target.value)}
                  >
                    {tiles.map((t) => (
                      <option key={t.id}>{t.id}</option>
                    ))}
                  </select>
                </label>
                {detail?.layers.map((l) => (
                  <button
                    key={l}
                    className={
                      layer === l ? "layer-button active" : "layer-button"
                    }
                    onClick={() => setLayer(l)}
                  >
                    {l === "rgb"
                      ? "Aerial RGB"
                      : l === "reference"
                        ? "Parcel reference"
                        : l === "height"
                          ? "Measured height"
                          : "Image support"}
                  </button>
                ))}
              </div>
              <div className="inspection-grid">
                <section className="panel map-panel">
                  {detail ? (
                    <>
                      <div className="section-title">
                        <h2>{tile}</h2>
                        <span className="status-pill">
                          {detail.split} · {detail.gsd_m} m/px
                        </span>
                      </div>
                      <div className="map-wrap tall">
                        <ImageMap
                          detail={detail}
                          source={
                            currentJob?.status === "completed" &&
                            currentJob.tile === tile &&
                            currentJob.dataset === dataset &&
                            page === "Inference"
                              ? jobAsset(currentJob.id)
                              : image
                          }
                          region={
                            page === "Review queue" ||
                            page === "Vision assistant"
                              ? region
                              : null
                          }
                        />
                        <div className="map-badge">
                          {layer === "height"
                            ? "Relative colour scale · metres"
                            : page === "Inference" &&
                                currentJob?.status === "completed"
                              ? "Amber = model prediction"
                              : "Reference layer ≠ prediction"}
                        </div>
                      </div>
                      <div className="panel-footer">
                        <span>
                          Drag to pan · scroll or use controls to zoom
                        </span>
                        <button
                          className="text-button"
                          onClick={() => {
                            setSelectedJob("");
                            setLayer("rgb");
                          }}
                        >
                          Reset to RGB
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="empty">Loading selected tile…</div>
                  )}
                </section>
                <aside className="inspection-side">
                  {page === "Inference" && (
                    <section className="panel">
                      <span className="eyebrow">MODEL EXECUTION</span>
                      <h2>Run & compare</h2>
                      <label>
                        Checkpoint
                        <select
                          value={checkpoint}
                          onChange={(e) => setCheckpoint(e.target.value)}
                        >
                          {catalog?.checkpoints.map((c) => (
                            <option key={c.id}>{c.id}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Compare with (optional)
                        <select
                          value={compare}
                          onChange={(e) => setCompare(e.target.value)}
                        >
                          <option value="">No comparison</option>
                          {catalog?.checkpoints
                            .filter((c) => c.id !== checkpoint)
                            .map((c) => (
                              <option key={c.id}>{c.id}</option>
                            ))}
                        </select>
                      </label>
                      <p className="muted">
                        Runs on real RGB + nDSM in overlapping windows. Boundary
                        threshold: 0.3 m. Training overlap is not verified.
                      </p>
                      <button
                        className="wide"
                        disabled={busy || !detail?.has_height || !checkpoint}
                        onClick={() => void launch()}
                      >
                        {busy ? "Submitting…" : "↗ Run inference"}
                      </button>
                      {!detail?.has_height && (
                        <p className="warning">
                          This tile has no paired nDSM. Choose a height-matched
                          dataset.
                        </p>
                      )}
                      <div className="divider" />
                      <h3>Completed on this tile</h3>
                      {completed.map((j) => (
                        <button
                          className={`result-select ${selectedJob === j.id ? "selected" : ""}`}
                          key={j.id}
                          onClick={() => setSelectedJob(j.id)}
                        >
                          <strong>{j.checkpoint?.split("/")[0]}</strong>
                          <span>
                            F1 {percent(j.metrics?.f1)} ·{" "}
                            {j.elapsed_seconds?.toFixed(1)} s
                          </span>
                        </button>
                      ))}
                      {!completed.length && (
                        <p className="muted">No completed inference yet.</p>
                      )}
                    </section>
                  )}
                  {page === "Evidence" && (
                    <section className="panel">
                      <span className="eyebrow">LAYER A / IMAGE SUPPORT</span>
                      <h2>Inspect the signal</h2>
                      {evidence ? (
                        <>
                          <div className="evidence-counts">
                            {Object.entries(evidence.counts).map(([k, v]) => (
                              <div key={k}>
                                <i className={k} />
                                <span>{k}</span>
                                <strong>{v.toLocaleString()}</strong>
                              </div>
                            ))}
                          </div>
                          <p className="muted">
                            Counts are reference-band pixels, not parcels.
                          </p>
                          <p>{evidence.method}</p>
                          <div className="message caution">
                            {evidence.warning}
                          </div>
                          <button
                            className="wide"
                            onClick={() => {
                              setLayer("evidence");
                              setPage("Review queue");
                            }}
                          >
                            Review candidate sections →
                          </button>
                        </>
                      ) : (
                        <p>Calculating image evidence…</p>
                      )}
                    </section>
                  )}
                  {(page === "Review queue" || page === "Vision assistant") && (
                    <section className="panel">
                      <span className="eyebrow">
                        {page === "Review queue"
                          ? "LAYER C / LABEL QA"
                          : "LAYER B / VISION ASSISTANCE"}
                      </span>
                      <h2>
                        {page === "Review queue"
                          ? "Review this section"
                          : "A second look"}
                      </h2>
                      <label>
                        Candidate crop
                        <select
                          value={region?.id || ""}
                          onChange={(e) =>
                            setRegion(
                              evidence?.candidates.find(
                                (c) => c.id === e.target.value,
                              ) || null,
                            )
                          }
                        >
                          {evidence?.candidates.map((c) => (
                            <option value={c.id} key={c.id}>
                              {c.id} · {percent(c.review_fraction)} weak /
                              uncertain
                            </option>
                          ))}
                        </select>
                      </label>
                      <p className="muted">
                        Candidates are ranked by weak or uncertain image
                        support. This does not establish label misalignment.
                      </p>
                      {page === "Review queue" ? (
                        <>
                          <label>
                            Reviewer note
                            <textarea
                              value={note}
                              onChange={(e) => setNote(e.target.value)}
                              placeholder="Describe the visible feature or suspected label issue…"
                            />
                          </label>
                          <div className="review-actions">
                            <button
                              disabled={busy || !region}
                              onClick={() => void saveReview("accept")}
                            >
                              Accept alignment
                            </button>
                            <button
                              className="secondary"
                              disabled={busy || !region}
                              onClick={() => void saveReview("reject")}
                            >
                              Flag for correction
                            </button>
                            <button
                              className="secondary"
                              disabled={busy || !region}
                              onClick={() => void saveReview("needs_survey")}
                            >
                              Needs survey evidence
                            </button>
                          </div>
                          <small>
                            Saved as a review record. No automatic label edits.
                          </small>
                        </>
                      ) : (
                        <>
                          <div className="provider">
                            <span
                              className={`status-pill ${catalog?.vision.configured ? "green" : "neutral"}`}
                            >
                              {catalog?.vision.configured
                                ? "Provider configured"
                                : "Provider not configured"}
                            </span>
                            <p>{catalog?.vision.provider}</p>
                            <small>
                              {catalog?.vision.daily_limit} requests/day · max
                              512 output tokens/request
                            </small>
                          </div>
                          <p className="muted">
                            Sends a 192px RGB crop and marked reference crop to
                            the configured provider. Charges depend on the
                            provider. Advice is not a survey decision.
                          </p>
                          <button
                            className="wide"
                            disabled={
                              busy || !region || !catalog?.vision.configured
                            }
                            onClick={() => void sendVision()}
                          >
                            {busy
                              ? "Requesting assessment…"
                              : "✧ Assess selected crop"}
                          </button>
                          {!catalog?.vision.configured && (
                            <p className="warning">
                              Set the vision key and model in the server
                              environment. No key is stored in this browser.
                            </p>
                          )}
                          {advice && (
                            <div className="advice">
                              <h3>Advisory assessment</h3>
                              <p>{advice}</p>
                            </div>
                          )}
                        </>
                      )}
                    </section>
                  )}
                  {detail && (
                    <section className="panel">
                      <span className="eyebrow">SOURCE RECORD</span>
                      <Details detail={detail} />
                    </section>
                  )}
                </aside>
              </div>
              {page === "Inference" && (
                <>
                  <Jobs
                    jobs={jobs.filter((j) => j.kind === "inference")}
                    onSelect={(j) => {
                      if (j.tile === tile && j.dataset === dataset)
                        setSelectedJob(j.id);
                      else
                        setNotice(
                          "Select the job’s dataset and tile to display its image.",
                        );
                    }}
                  />
                  {currentJob?.metrics && (
                    <section className="panel">
                      <div className="section-title">
                        <h2>Measured reference agreement</h2>
                        <a
                          className="download"
                          href={jobAsset(currentJob.id, "report")}
                          download
                        >
                          Download report ↓
                        </a>
                      </div>
                      <div className="stats-grid">
                        {(["precision", "recall", "f1", "iou"] as const).map(
                          (k) => (
                            <Stat
                              key={k}
                              label={k.toUpperCase()}
                              value={percent(currentJob.metrics?.[k])}
                              detail="Stitched valid pixels"
                            />
                          ),
                        )}
                      </div>
                      <p className="muted">
                        {currentJob.evaluation_scope}{" "}
                        {currentJob.metrics.definition}
                      </p>
                      <a href={jobAsset(currentJob.id, "arrays")} download>
                        Download numerical predictions (.npz)
                      </a>
                    </section>
                  )}
                </>
              )}
              {(page === "Review queue" || page === "Vision assistant") && (
                <section className="panel">
                  <h2>Review history</h2>
                  {reviews
                    .filter((r) => r.tile === tile && r.dataset === dataset)
                    .map((r) => (
                      <div className="review-record" key={r.id}>
                        <span className="status-pill">{r.decision}</span>
                        <strong>{r.region || "Vision assessment"}</strong>
                        <p>{r.note || r.advice || "No note supplied"}</p>
                        <small>
                          {new Date(r.created * 1000).toLocaleString()}
                        </small>
                      </div>
                    ))}
                  {!reviews.some(
                    (r) => r.tile === tile && r.dataset === dataset,
                  ) && <p className="muted">No saved reviews for this tile.</p>}
                </section>
              )}
            </>
          )}
          {page === "Training" && (
            <>
              <div className="training-layout">
                <section className="panel">
                  <span className="eyebrow">EXPERIMENT RECORD</span>
                  <h2>Learning, measured.</h2>
                  <label>
                    Recorded run
                    <select
                      value={run}
                      onChange={(e) => setRun(e.target.value)}
                    >
                      {training.map((t) => (
                        <option key={t.id}>{t.id}</option>
                      ))}
                    </select>
                  </label>
                  {currentRun && (
                    <>
                      <div className="stats-grid three">
                        <Stat
                          label="RECORDED EPOCHS"
                          value={String(currentRun.epochs)}
                          detail={currentRun.scope}
                        />
                        <Stat
                          label="LATEST VALIDATION F1"
                          value={percent(last?.validation?.f1)}
                          detail="Within this run’s evaluator"
                        />
                        <Stat
                          label="LATEST TRAIN LOSS"
                          value={last?.train_loss?.toFixed(3) || "—"}
                          detail="Do not compare different objectives"
                        />
                      </div>
                      <Sparkline history={currentRun.history} />
                      {currentRun.result && (
                        <div className="message caution">
                          Same-image fit experiment: model A F1{" "}
                          {percent(
                            currentRun.result.model_a_sdf_regression?.f1,
                          )}
                          , model B F1{" "}
                          {percent(
                            currentRun.result.model_b_direct_classifier?.f1,
                          )}
                          . This is not held-out accuracy.
                        </div>
                      )}
                      <p className="muted">
                        {currentRun.status}. Existing evaluations may use
                        different loss definitions or patch aggregation. Check
                        the experiment before comparing scores.
                      </p>
                      <div className="table-scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>Epoch</th>
                              <th>Train loss</th>
                              <th>Validation loss</th>
                              <th>F1</th>
                              <th>Duration</th>
                            </tr>
                          </thead>
                          <tbody>
                            {currentRun.history
                              .slice(-10)
                              .reverse()
                              .map((e) => (
                                <tr key={e.epoch}>
                                  <td>{e.epoch}</td>
                                  <td>{e.train_loss?.toFixed(3)}</td>
                                  <td>{e.validation?.loss?.toFixed(3)}</td>
                                  <td>{percent(e.validation?.f1)}</td>
                                  <td>{e.seconds?.toFixed(1)} s</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </section>
                <section className="panel">
                  <span className="eyebrow">CONTROLLED EXPERIMENT</span>
                  <h2>Start a real-data run</h2>
                  <p className="muted">
                    Uses the selected dataset and existing real-image trainer. A
                    fresh model is created. GPU is used only if available to the
                    server.
                  </p>
                  <label>
                    Total epochs
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Batch size
                    <input
                      type="number"
                      min="1"
                      max="16"
                      value={batch}
                      onChange={(e) => setBatch(Number(e.target.value))}
                    />
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={visibilityAware}
                      onChange={(e) => setVisibilityAware(e.target.checked)}
                    />
                    Visibility-aware boundary loss
                  </label>
                  <p className="muted">
                    Down-weights the boundary loss term by the same RGB/nDSM
                    edge-evidence score the Evidence tab shows a reviewer, so a
                    boundary with no visible image feature is penalized less.
                    Off reproduces the trainer's exact prior loss.
                  </p>
                  <button
                    disabled={busy || !dataset}
                    onClick={() =>
                      void guard(async () => {
                        await request("/training", {
                          dataset,
                          epochs,
                          batch_size: batch,
                          visibility_aware: visibilityAware,
                        });
                        setNotice(
                          "Training queued. Epoch history updates when the trainer writes it.",
                        );
                        await refresh();
                      })
                    }
                  >
                    Start training run ↗
                  </button>
                  <p className="warning">
                    Use a verified dataset snapshot. Missing inputs or
                    in-progress collections are refused. Runs have a six-hour
                    execution limit.
                  </p>
                </section>
              </div>
              <Jobs
                jobs={jobs.filter((j) => j.kind === "training")}
                onSelect={(j) => {
                  if (j.run) setRun(j.run);
                }}
              />
            </>
          )}
          {page === "Ward operations" && (
            <div className="legacy-console">
              <OperatorConsole />
            </div>
          )}
          <footer className="footer">
            <span>GeoCadastra · Research workspace</span>
            <span>Measured outputs. Explicit uncertainty. Human review.</span>
          </footer>
        </div>
      </main>
    </div>
    </div>
  );
}
function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
function Details({ detail: d }: { detail: Detail }) {
  return (
    <>
      <dl className="details">
        <dt>Coordinate system</dt>
        <dd>{d.crs || "Not recorded"}</dd>
        <dt>Image resolution</dt>
        <dd>{d.gsd_m} m / pixel</dd>
        <dt>Dimensions</dt>
        <dd>
          {d.width} × {d.height}
        </dd>
        <dt>Dataset split</dt>
        <dd>{d.split}</dd>
        <dt>Height data</dt>
        <dd>{d.has_height ? "Measured nDSM" : "Unavailable"}</dd>
        <dt>Capture date</dt>
        <dd>
          {d.capture
            ? new Date(d.capture).toLocaleDateString()
            : "See source metadata"}
        </dd>
        {d.height_range_m && (
          <>
            <dt>Height P05–P95</dt>
            <dd>{d.height_range_m.map((v) => v.toFixed(1)).join("–")} m</dd>
          </>
        )}
      </dl>
      {d.sha256 && (
        <details>
          <summary>Provenance</summary>
          <code className="hash">SHA256 {d.sha256}</code>
          {d.source && (
            <a href={d.source} target="_blank" rel="noreferrer">
              Open source ↗
            </a>
          )}
          <p className="muted">
            {d.limitations?.join(" · ") ||
              "Reference labels are not independently surveyed ground truth."}
          </p>
        </details>
      )}
    </>
  );
}
function Jobs({ jobs, onSelect }: { jobs: Job[]; onSelect: (j: Job) => void }) {
  return (
    <section className="panel jobs-panel">
      <div className="section-title">
        <div>
          <span className="eyebrow">EXECUTION LOG</span>
          <h2>Workspace jobs</h2>
        </div>
        <span className="muted">Updates every 10 seconds</span>
      </div>
      {jobs.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Source / run</th>
                <th>Checkpoint / task</th>
                <th>Status</th>
                <th>Measured time</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td>
                    <strong>{j.tile || j.run || j.dataset}</strong>
                    <small>{new Date(j.created * 1000).toLocaleString()}</small>
                  </td>
                  <td>{j.checkpoint || j.kind}</td>
                  <td>
                    <span
                      className={`status-pill ${j.status === "completed" ? "green" : j.status === "failed" ? "red" : "neutral"}`}
                    >
                      {j.status}
                    </span>
                    {j.error && <p className="error-text">{j.error}</p>}
                  </td>
                  <td>
                    {j.elapsed_seconds === undefined
                      ? "—"
                      : `${j.elapsed_seconds.toFixed(1)} s`}
                  </td>
                  <td>
                    <button className="text-button" onClick={() => onSelect(j)}>
                      Inspect ↗
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty compact">
          No workspace jobs yet. Run an inference or start a controlled
          experiment.
        </div>
      )}
    </section>
  );
}
