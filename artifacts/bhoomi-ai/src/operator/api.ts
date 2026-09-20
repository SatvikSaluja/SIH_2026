// Thin fetch wrapper matching geocadastra/api/main.py's actual request/response
// shapes -- every field here was read from that file, not guessed from a REST
// convention. If a shape below stops matching the backend, the backend moved,
// not this file.

// Was VITE_API_BASE pointing straight at uvicorn with CORS, back when this
// console was its own Vite app. In the unified app everything reaches the
// backend the same way -- relative /api/*, through the Express proxy -- so
// there is one access pattern, not two. The panels below are otherwise
// unchanged from that app; the proxy's allowlist is what decides which of
// these endpoints are actually reachable from a browser.
const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${init?.method ?? "GET"} ${path} -> ${res.status}: ${body}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export interface IngestRequest {
  seed: number;
  width?: number;
  height?: number;
  gsd?: number;
  n_arterial_h?: number;
  n_arterial_v?: number;
}
export interface IngestResponse {
  ward_job_id: number;
  n_blocks: number;
  n_parcels: number;
}

export interface BlockStatus {
  block_id: number;
  status: string; // pending | running | done | failed
  error: string | null;
}
export interface WardStatus {
  ward_job_id: number;
  status: string;
  blocks: BlockStatus[];
}

export interface ParcelFeature {
  face_id: number;
  block_id: number;
  parcel_id: number | null;
  geometry: GeoJSON.Geometry;
}

export interface ConflictRow {
  id: number;
  block_id: number;
  node_id: number;
  sources: string[];
  disagreement_m: number | null;
  kind: string;
  detail: Record<string, unknown>;
  geometry: GeoJSON.Geometry;
}

export interface ParcelAreaRow {
  parcel_id: number;
  style: string;
  recorded_area_m2: number;
  predicted_area_m2: number;
  error_m2: number;
  tolerance_m2: number;
  face_ids: number[];
  within_tolerance: boolean;
}
export interface BlockConstraintReport {
  block_id: number;
  parcels: ParcelAreaRow[];
  unassigned_face_ids: number[];
  block_coverage_error_m2: number | null;
  block_coverage_tolerance_m2: number;
  constraints_satisfied: boolean;
}
export interface ConstraintsResponse {
  recorded_area_constraints_satisfied: boolean;
  boundary_certification: string; // "not_calibrated" today, always
  blocks: BlockConstraintReport[];
}

export interface ErrorSummary {
  n: number;
  p50: number | null;
  p90: number | null;
}
export interface ParcelCountError {
  predicted: number;
  recorded: number;
  over_segmentation: number;
  under_segmentation: number;
}
export interface StratumAnalytics {
  boundary_position_error_p50: number | null;
  boundary_position_error_p90: number | null;
  evaluation_population: string;
  fusion_control_residual: ErrorSummary;
  topology_validity_rate: number | null;
  parcel_count: ParcelCountError;
  face_count: number;
  topology_scope: string;
  area_error_relative: ErrorSummary;
  area_constraints: { within_tolerance: number; total: number; missing_parcel_ids: number[] };
  n_gt_points: number;
}
export type AnalyticsResponse = Record<string, StratumAnalytics>;

export interface EditRequest {
  block_id: number;
  node_id: number;
  x: number;
  y: number;
  author?: string;
}

export interface FieldVerificationRequest extends EditRequest {
  parcel_id: number;
}

export interface CoRegisterRequest {
  control_points: [number, number, number, number][]; // src_x, src_y, dst_x, dst_y
}
export interface CoRegisterResponse {
  transform: [number[], number[]];
  residual_m: number;
}

export const api = {
  ingest: (body: IngestRequest) =>
    req<IngestResponse>("/wards/ingest", { method: "POST", body: JSON.stringify(body) }),

  run: (wardJobId: number) =>
    req<WardStatus>(`/wards/${wardJobId}/run`, { method: "POST" }),

  status: (wardJobId: number) => req<WardStatus>(`/wards/${wardJobId}/status`),

  parcels: (wardJobId: number, bbox: [number, number, number, number]) => {
    const [minx, miny, maxx, maxy] = bbox;
    return req<{ parcels: ParcelFeature[] }>(
      `/wards/${wardJobId}/parcels?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}`
    );
  },

  conflicts: (wardJobId: number) =>
    req<{ conflicts: ConflictRow[] }>(`/wards/${wardJobId}/conflicts`),

  constraints: (wardJobId: number) => req<ConstraintsResponse>(`/wards/${wardJobId}/constraints`),

  analytics: (wardJobId: number) => req<AnalyticsResponse>(`/wards/${wardJobId}/analytics`),

  edit: (wardJobId: number, body: EditRequest) =>
    req<{ status: string }>(`/wards/${wardJobId}/edit`, { method: "POST", body: JSON.stringify(body) }),

  fieldVerification: (wardJobId: number, body: FieldVerificationRequest) =>
    req<{ status: string }>(`/wards/${wardJobId}/field-verification`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  coregister: (wardJobId: number, body: CoRegisterRequest) =>
    req<CoRegisterResponse>(`/wards/${wardJobId}/coregister`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  parcelAssociations: (
    wardJobId: number,
    body: { block_id: number; assignments: Record<number, number>; author?: string }
  ) =>
    req<{ status: string }>(`/wards/${wardJobId}/parcel-associations`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  tileUrl: (wardJobId: number) => `${BASE}/wards/${wardJobId}/tiles/{z}/{x}/{y}.mvt`,
};
