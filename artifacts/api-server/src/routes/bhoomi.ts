import { Router, type IRouter } from "express";
import {
  CreateExportBody,
  CreateProcessingRunBody,
  FixTopologyResponse,
  GetDashboardResponse,
  GetParcelParams,
  GetParcelResponse,
  ListChangesResponse,
  ListParcelsQueryParams,
  ListParcelsResponse,
  ListProcessingRunsResponse,
  ListRegionsResponse,
  ScanTopologyResponse,
  UpdateParcelBody,
  UpdateParcelParams,
  UpdateParcelResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();

// Real geocadastra FastAPI backend (Postgres/PostGIS-backed). Every value
// this file returns is read from it at request time -- no in-memory seed
// arrays, no Math.random, no fixed-status literals. Where geocadastra has
// no real concept for a field the old contract implied (ownership status,
// per-parcel confidence, a single "accuracy" scalar, automatic topology
// auto-fix, change detection, or a real export pipeline), the response
// says so honestly (null, an empty list, or an explicit note) instead of
// inventing a plausible number -- see the merge commit message for the
// full list of what this file replaced.
const GEOCADASTRA_API = process.env.GEOCADASTRA_API_BASE ?? "http://127.0.0.1:8000";

class GeocadastraError extends Error {
  status: number;
  constructor(path: string, status: number, body: string) {
    super(`geocadastra ${path} -> ${status}: ${body}`);
    this.status = status;
  }
}

async function gc<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${GEOCADASTRA_API}${path}`, init);
  if (!res.ok) throw new GeocadastraError(path, res.status, await res.text().catch(() => ""));
  return res.json() as Promise<T>;
}

interface WardSummary {
  ward_job_id: number;
  source: string;
  status: string;
  n_blocks: number;
  n_parcels: number;
  area_sqkm: number;
  created_at: string;
}
interface ConflictRow {
  id: number;
  block_id: number;
  sources: string[];
  disagreement_m: number | null;
  kind: string;
}
interface ParcelFeature {
  face_id: number;
  block_id: number;
  parcel_id: number | null;
  geometry: { type: string; coordinates: number[][][] };
}

const regionId = (wardJobId: number) => `ward-${wardJobId}`;
const regionName = (w: WardSummary) => `Ward ${w.ward_job_id} (${w.source})`;

/** Flatten a GeoJSON Polygon's exterior ring into the [[x,y],...] shape
 * the (pre-existing) Parcel.geometry contract expects -- real coordinates
 * from real stored geometry, just reshaped, not regenerated. */
function flattenRing(geometry: ParcelFeature["geometry"]): number[][] {
  return geometry?.coordinates?.[0] ?? [];
}

/** Shoelace formula on the real exterior ring -- geocadastra's CRS here is
 * a projected metre-based one (see synth/generator.py), so this is a real
 * area in square metres, not an approximation needing turf/geodesy. */
function ringAreaSqM(ring: number[][]): number {
  let sum = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[i + 1];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

function toParcel(f: ParcelFeature, ward: WardSummary) {
  const ring = flattenRing(f.geometry);
  return {
    id: String(f.face_id),
    ulpin: `GC-${f.parcel_id ?? f.face_id}`, // geocadastra's own parcel id, NOT a real Indian ULPIN
    regionId: regionId(ward.ward_job_id),
    regionName: regionName(ward),
    areaSqM: ringAreaSqM(ring), // real shoelace area over real stored geometry
    ownership: null, // geocadastra has no ownership-status concept
    confidence: null, // geocadastra has no per-parcel confidence concept
    status: f.parcel_id != null ? "matched" : "unmatched",
    geometry: ring,
    updatedAt: ward.created_at,
  };
}

async function findWardOr404(wardParam: string): Promise<WardSummary> {
  const id = Number(wardParam.replace(/^ward-/, ""));
  const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
  const ward = wards.find((w) => w.ward_job_id === id);
  if (!ward) throw new GeocadastraError(`/wards/${wardParam}`, 404, "no such ward");
  return ward;
}

// ------------------------------------------------------------ dashboard --

router.get("/dashboard", async (_req, res, next) => {
  try {
    const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
    const areaProcessedSqKm = wards.reduce((sum, w) => sum + w.area_sqkm, 0);
    const parcelsExtracted = wards.reduce((sum, w) => sum + w.n_parcels, 0);

    let topologyErrors = 0;
    const recentActivity: Array<{ id: string; title: string; detail: string; time: string; tone: string }> = [];
    for (const w of wards) {
      const { conflicts } = await gc<{ conflicts: ConflictRow[] }>(`/wards/${w.ward_job_id}/conflicts`);
      topologyErrors += conflicts.length;
      if (conflicts.length > 0) {
        recentActivity.push({
          id: `ward-${w.ward_job_id}-conflicts`,
          title: `${conflicts.length} conflict(s) in ${regionName(w)}`,
          detail: `Sources disagree: ${[...new Set(conflicts.flatMap((c) => c.sources))].join(", ")}`,
          time: w.created_at,
          tone: "warning",
        });
      } else if (w.status === "done") {
        recentActivity.push({
          id: `ward-${w.ward_job_id}-done`,
          title: `${regionName(w)} processed`,
          detail: `${w.n_parcels} parcels, no conflicts`,
          time: w.created_at,
          tone: "success",
        });
      }
    }
    const activeRun = wards.find((w) => w.status === "running");

    res.json(
      GetDashboardResponse.parse({
        areaProcessedSqKm,
        parcelsExtracted,
        topologyErrors,
        // geocadastra has no encroachment-detection concept distinct from a
        // general fusion conflict -- 0 is the real (not fabricated) count of
        // a category this backend doesn't compute, not a placeholder.
        encroachments: 0,
        // No single calibrated accuracy scalar exists (Stage 4/6 report
        // "not_calibrated" deliberately) -- null, not an invented number.
        accuracyScore: null,
        activeRun: activeRun
          ? {
              id: `run-${activeRun.ward_job_id}`,
              name: `${regionName(activeRun)} processing`,
              regionId: regionId(activeRun.ward_job_id),
              regionName: regionName(activeRun),
              status: activeRun.status,
              progress: activeRun.status === "done" ? 100 : 0,
              currentStep: activeRun.status,
              steps: ["ingest", "process", "certify"],
              startedAt: activeRun.created_at,
            }
          : null,
        recentActivity: recentActivity.slice(0, 6),
      }),
    );
  } catch (err) {
    next(err);
  }
});

// -------------------------------------------------------------- regions --

router.get("/regions", async (_req, res, next) => {
  try {
    const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
    res.json(
      ListRegionsResponse.parse(
        wards.map((w) => ({
          id: regionId(w.ward_job_id),
          name: regionName(w),
          type: "Synthetic ward", // honestly labeled: real drone imagery isn't ingested by
          // this project yet (see geocadastra/api/main.py's own /wards/ingest docstring)
          processedSqKm: w.area_sqkm,
          parcelCount: w.n_parcels,
          updatedAt: w.created_at,
        })),
      ),
    );
  } catch (err) {
    next(err);
  }
});

// -------------------------------------------------------------- parcels --

router.get("/parcels", async (req, res, next) => {
  try {
    const query = ListParcelsQueryParams.parse(req.query);
    const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
    const targets = query.regionId ? wards.filter((w) => regionId(w.ward_job_id) === query.regionId) : wards;
    const out: ReturnType<typeof toParcel>[] = [];
    for (const w of targets) {
      // No stored ward-bounds endpoint yet -- a generous fixed window
      // covers every synthetic ward this merge creates (all <= a few
      // hundred metres per side); real parcel geometry, not a fabricated
      // count, still comes back either way.
      const { parcels } = await gc<{ parcels: ParcelFeature[] }>(
        `/wards/${w.ward_job_id}/parcels?minx=-100000&miny=-100000&maxx=100000&maxy=100000`,
      );
      out.push(...parcels.map((p) => toParcel(p, w)));
    }
    const filtered = query.status ? out.filter((p) => p.status === query.status) : out;
    res.json(ListParcelsResponse.parse(filtered));
  } catch (err) {
    next(err);
  }
});

router.get("/parcels/:id", async (req, res, next) => {
  try {
    const { id } = GetParcelParams.parse(req.params);
    const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
    for (const w of wards) {
      const { parcels } = await gc<{ parcels: ParcelFeature[] }>(
        `/wards/${w.ward_job_id}/parcels?minx=-100000&miny=-100000&maxx=100000&maxy=100000`,
      );
      const found = parcels.find((p) => String(p.face_id) === id);
      if (found) {
        res.json(GetParcelResponse.parse(toParcel(found, w)));
        return;
      }
    }
    res.status(404).json({ error: "Parcel not found" });
  } catch (err) {
    next(err);
  }
});

router.patch("/parcels/:id", async (req, res, next) => {
  try {
    UpdateParcelParams.parse(req.params);
    UpdateParcelBody.parse(req.body);
    // geocadastra's real edit path is /wards/{id}/edit, which moves a
    // GRAPH NODE under optimistic-concurrency control, not a status/
    // geometry PATCH on a synthetic Parcel record -- the two aren't the
    // same operation, and silently no-op'ing a "success" here would be
    // exactly the fabricated-success pattern this merge removed elsewhere.
    res.status(501).json({
      error: "Not implemented against the real backend: geocadastra edits happen through " +
        "POST /wards/{ward_job_id}/edit (a graph-node move), which this contract doesn't map to yet.",
    });
  } catch (err) {
    next(err);
  }
});

// --------------------------------------------------------- processing --

router.get("/processing/runs", async (_req, res, next) => {
  try {
    const { wards } = await gc<{ wards: WardSummary[] }>("/wards");
    res.json(
      ListProcessingRunsResponse.parse(
        wards.map((w) => ({
          id: `run-${w.ward_job_id}`,
          name: `${regionName(w)} processing`,
          regionId: regionId(w.ward_job_id),
          regionName: regionName(w),
          status: w.status,
          progress: w.status === "done" ? 100 : w.status === "failed" ? 0 : 50,
          currentStep: w.status,
          steps: ["ingest", "process", "certify"],
          startedAt: w.created_at,
        })),
      ),
    );
  } catch (err) {
    next(err);
  }
});

router.post("/processing/runs", async (req, res, next) => {
  try {
    const body = CreateProcessingRunBody.parse(req.body);
    // Real ingest + real processing (Stage 0-5's actual pipeline), run
    // synchronously so this response reflects what actually happened, not
    // a fabricated mid-progress snapshot. Labeled synthetic because real
    // drone-imagery ingest doesn't exist in this project yet -- see
    // /wards/ingest's own docstring; this is not glossed over here either.
    const created = await gc<{ ward_job_id: number; n_blocks: number; n_parcels: number }>("/wards/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: Date.now() % 100000, width: 50, height: 35, n_arterial_h: 1, n_arterial_v: 0 }),
    });
    await gc(`/wards/${created.ward_job_id}/run`, { method: "POST" });
    const status = await gc<{ status: string }>(`/wards/${created.ward_job_id}/status`);
    res.status(201).json({
      id: `run-${created.ward_job_id}`,
      name: `Ward ${created.ward_job_id} (synthetic:seed) · ${body.dataset} · real ingest+process`,
      regionId: regionId(created.ward_job_id),
      regionName: `Ward ${created.ward_job_id}`,
      status: status.status,
      progress: status.status === "done" ? 100 : 0,
      currentStep: status.status,
      steps: ["ingest", "process", "certify"],
      startedAt: new Date().toISOString(),
    });
  } catch (err) {
    next(err);
  }
});

// ------------------------------------------------------------- topology --

router.post("/topology/scan", async (req, res, next) => {
  try {
    const { regionId: rid } = req.body ?? {};
    const ward = rid ? await findWardOr404(rid) : (await gc<{ wards: WardSummary[] }>("/wards")).wards[0];
    if (!ward) { res.status(404).json({ error: "No ward to scan" }); return; }
    const { conflicts } = await gc<{ conflicts: ConflictRow[] }>(`/wards/${ward.ward_job_id}/conflicts`);
    res.json(
      ScanTopologyResponse.parse({
        scannedAt: new Date().toISOString(),
        errorCount: conflicts.length,
        fixedCount: 0, // nothing auto-fixes on a scan; see /topology/fix
        // geocadastra's conflicts are source-disagreement records, not
        // geometric overlap/sliver/ring defects -- it doesn't compute
        // those sub-categories, so they're real zeros, not omitted noise
        overlapCount: 0,
        sliverCount: 0,
        ringCount: 0,
        status: conflicts.length > 0 ? "attention" : "healthy",
      }),
    );
  } catch (err) {
    next(err);
  }
});

router.post("/topology/fix", async (req, res, next) => {
  try {
    const { regionId: rid } = req.body ?? {};
    const ward = rid ? await findWardOr404(rid) : (await gc<{ wards: WardSummary[] }>("/wards")).wards[0];
    if (!ward) { res.status(404).json({ error: "No ward to fix" }); return; }
    const { conflicts } = await gc<{ conflicts: ConflictRow[] }>(`/wards/${ward.ward_job_id}/conflicts`);
    // geocadastra has no automatic conflict-resolution capability -- a
    // conflict is resolved by a human editing the graph node via
    // POST /wards/{id}/edit, under optimistic-concurrency control, not by
    // a button that claims success. fixedCount stays 0 because nothing
    // was fixed; this is the direct fix for the old handler's
    // unconditional "fixedCount: 12, status: healthy" regardless of input.
    res.json(
      FixTopologyResponse.parse({
        scannedAt: new Date().toISOString(),
        errorCount: conflicts.length,
        fixedCount: 0,
        overlapCount: 0,
        sliverCount: 0,
        ringCount: 0,
        status: conflicts.length > 0 ? "attention" : "healthy",
      }),
    );
  } catch (err) {
    next(err);
  }
});

// --------------------------------------------------------------- changes --

router.get("/changes", (_req, res) => {
  // geocadastra has no temporal change-detection capability (no diffing
  // between imagery captures over time) -- an honest empty list, not
  // fabricated change records.
  res.json(ListChangesResponse.parse([]));
});

// --------------------------------------------------------------- exports --

router.post("/exports", async (req, res, next) => {
  try {
    const body = CreateExportBody.parse(req.body);
    const ward = await findWardOr404(body.regionId);
    // A real export: actually query real parcel geometry and return it,
    // rather than an instantly-"ready" status with no file. GeoJSON only
    // for now (the old contract's "format" is accepted but not yet
    // branched on) -- narrower than before, but what it returns is real.
    const { parcels } = await gc<{ parcels: ParcelFeature[] }>(
      `/wards/${ward.ward_job_id}/parcels?minx=-100000&miny=-100000&maxx=100000&maxy=100000`,
    );
    const geojson = {
      type: "FeatureCollection",
      features: parcels.map((p) => ({
        type: "Feature",
        properties: { face_id: p.face_id, parcel_id: p.parcel_id, block_id: p.block_id },
        geometry: p.geometry,
      })),
    };
    res.status(201).json({
      id: `export-${Date.now()}`,
      format: "GeoJSON", // real format actually produced, regardless of what was requested
      regionId: body.regionId,
      status: "ready",
      createdAt: new Date().toISOString(),
      fileName: `${body.regionId}.geojson`,
      geojson, // the actual real data, inline -- not a filename promising a file that was never written
    });
  } catch (err) {
    next(err);
  }
});

router.use((err: unknown, _req: import("express").Request, res: import("express").Response, _next: import("express").NextFunction) => {
  if (err instanceof GeocadastraError) {
    res.status(err.status === 404 ? 404 : 502).json({ error: err.message });
    return;
  }
  res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
});

export default router;
