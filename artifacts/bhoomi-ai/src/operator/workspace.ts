// Same repoint as operator/api.ts: relative /api/*, through the Express
// proxy, instead of the old standalone app's direct-to-uvicorn CORS base.
export const API = "/api";
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    API + "/workspace" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    let reason = await response.text();
    try {
      reason = JSON.parse(reason).detail;
    } catch {
      /* plain response */
    }
    throw new Error(String(reason));
  }
  return response.json();
}
export interface Dataset {
  id: string;
  count: number;
  status: string;
  crs: string;
  splits: Record<string, number>;
  source?: string;
  limitations: string[];
}
export interface Checkpoint {
  id: string;
  bytes: number;
  scope: string;
}
export interface Catalog {
  datasets: Dataset[];
  checkpoints: Checkpoint[];
  vision: {
    configured: boolean;
    model: string | null;
    daily_limit: number;
    provider: string;
  };
  certification: string;
}
export interface Tile {
  id: string;
  split: string;
  gsd_m: number;
  parcels?: number;
  valid_fraction?: number;
}
export interface Detail extends Tile {
  dataset: string;
  width: number;
  height: number;
  crs: string;
  layers: string[];
  has_height: boolean;
  capture?: string;
  source?: string;
  sha256?: string;
  height_range_m?: number[];
  limitations: string[];
}
export interface Candidate {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  review_fraction: number;
  reference_pixels: number;
}
export interface Evidence {
  counts: Record<string, number>;
  candidates: Candidate[];
  method: string;
  warning: string;
}
export interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  iou: number;
  definition: string;
}
export interface Job {
  id: string;
  kind: string;
  status: string;
  created: number;
  dataset: string;
  tile?: string;
  checkpoint?: string;
  elapsed_seconds?: number;
  metrics?: Metrics;
  error?: string;
  run?: string;
  evaluation_scope?: string;
  checkpoint_sha256?: string;
  input_sha256?: string;
}
export interface Review {
  dataset: string;
  id: string;
  tile: string;
  region?: string;
  decision: string;
  note?: string;
  advice?: string;
  created: number;
}
export interface Epoch {
  epoch: number;
  train_loss: number;
  validation: Partial<Metrics> & { loss: number; distance_mae_m?: number };
  seconds: number;
}
export interface Training {
  id: string;
  history: Epoch[];
  epochs: number;
  status: string;
  scope: string;
  checkpoints: string[];
  result?: {
    model_a_sdf_regression: Metrics;
    model_b_direct_classifier: Metrics;
  };
}
export const asset = (
  dataset: string,
  tile: string,
  layer = "rgb",
  thumb = false,
) =>
  `${API}/workspace/tiles/${encodeURIComponent(dataset)}/${encodeURIComponent(tile)}/image?layer=${layer}&thumb=${thumb}`;
export const jobAsset = (id: string, kind = "image") =>
  `${API}/workspace/jobs/${id}/artifact?kind=${kind}`;
