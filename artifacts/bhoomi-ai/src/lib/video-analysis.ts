export type ExtractedFrame = {
  data: string;
  mimeType: string;
  timestampSeconds: number;
  frameIndex: number;
};

export type FrameObservation = {
  label: string;
  category: string;
  confidence: number;
  evidenceNote: string;
  timestampSeconds: number | null;
  boundingBox: [number, number, number, number];
  frameIndex?: number;
};

const ANALYZE_SIZE = 96;
const BLOCK = 8;
const GRID = ANALYZE_SIZE / BLOCK;
const MAX_DECODE_FRAMES = 600;
const TARGET_FPS = 30;

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const target = clamp(time, 0, Math.max(0, duration - 0.04));
    if (Math.abs(video.currentTime - target) < 0.012 && video.readyState >= 2) {
      resolve();
      return;
    }
    const finish = () => {
      video.removeEventListener('seeked', finish);
      video.removeEventListener('error', finish);
      window.clearTimeout(timer);
      resolve();
    };
    const timer = window.setTimeout(finish, 1800);
    video.addEventListener('seeked', finish);
    video.addEventListener('error', finish);
    video.currentTime = target;
  });
}

function loadVideo(file: File): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';
    const url = URL.createObjectURL(file);
    video.src = url;

    const cleanupFail = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Unable to decode video'));
    };

    video.onerror = cleanupFail;
    video.onloadedmetadata = () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        cleanupFail();
        return;
      }
      const ready = () => resolve(video);
      if (video.readyState >= 2) ready();
      else video.onloadeddata = ready;
    };
  });
}

function buildTimestamps(duration: number): number[] {
  const safeDuration = Math.max(0.04, duration);
  const nativeCount = Math.max(1, Math.round(safeDuration * TARGET_FPS));
  const sampleCount = Math.min(MAX_DECODE_FRAMES, nativeCount);
  const timestamps: number[] = [];
  if (sampleCount === 1) {
    timestamps.push(Math.min(0.02, safeDuration / 2));
    return timestamps;
  }
  for (let i = 0; i < sampleCount; i++) {
    const t = (i / (sampleCount - 1)) * Math.max(0, safeDuration - 0.04);
    timestamps.push(Number(t.toFixed(3)));
  }
  return timestamps;
}

type Region = {
  row: number;
  col: number;
  avgR: number;
  avgG: number;
  avgB: number;
  bright: number;
  sat: number;
  variance: number;
  veg: number;
  struct: number;
  hydro: number;
  road: number;
};

function regionStats(pixels: Uint8ClampedArray, size: number): Region[] {
  const regions: Region[] = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let count = 0;
      const samples: number[] = [];
      for (let py = 0; py < BLOCK; py++) {
        for (let px = 0; px < BLOCK; px++) {
          const x = c * BLOCK + px;
          const y = r * BLOCK + py;
          const idx = (y * size + x) * 4;
          const red = pixels[idx];
          const green = pixels[idx + 1];
          const blue = pixels[idx + 2];
          sumR += red;
          sumG += green;
          sumB += blue;
          samples.push((red + green + blue) / 3);
          count++;
        }
      }
      const avgR = sumR / count;
      const avgG = sumG / count;
      const avgB = sumB / count;
      const bright = (avgR + avgG + avgB) / 3;
      const maxC = Math.max(avgR, avgG, avgB);
      const minC = Math.min(avgR, avgG, avgB);
      const sat = maxC === 0 ? 0 : (maxC - minC) / maxC;
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

      const exg = 2 * avgG - avgR - avgB;
      const ndviLike = (avgG - avgR) / (avgG + avgR + 1);
      const veg = exg > 18 && ndviLike > 0.04 && avgG > 40 ? exg + ndviLike * 80 : 0;
      const hydro =
        (avgB > avgR + 10 && bright < 120) || (bright < 55 && sat < 0.22) ? (130 - bright) + Math.max(0, avgB - avgR) : 0;
      const struct =
        veg === 0 && variance > 180 && ((bright > 145 && sat < 0.28) || (avgR > avgG + 12 && avgR > 110))
          ? variance * 0.35 + bright * 0.2
          : 0;
      const road =
        veg === 0 && hydro === 0 && bright >= 70 && bright <= 175 && sat < 0.2 && variance < 420
          ? 180 - sat * 400
          : 0;

      regions.push({ row: r, col: c, avgR, avgG, avgB, bright, sat, variance, veg, struct, hydro, road });
    }
  }
  return regions;
}

type Cluster = { rows: number[]; cols: number[]; score: number; regions: Region[] };

function clusterByScore(regions: Region[], key: keyof Pick<Region, 'veg' | 'struct' | 'hydro' | 'road'>, minScore: number): Cluster[] {
  const mask = Array.from({ length: GRID }, () => Array<boolean>(GRID).fill(false));
  const map = new Map<string, Region>();
  for (const region of regions) {
    map.set(`${region.row}:${region.col}`, region);
    if (region[key] >= minScore) mask[region.row][region.col] = true;
  }

  const visited = Array.from({ length: GRID }, () => Array<boolean>(GRID).fill(false));
  const clusters: Cluster[] = [];

  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      if (!mask[r][c] || visited[r][c]) continue;
      const stack = [[r, c]];
      visited[r][c] = true;
      const rows: number[] = [];
      const cols: number[] = [];
      const clusterRegions: Region[] = [];
      let score = 0;
      while (stack.length) {
        const [cr, cc] = stack.pop()!;
        rows.push(cr);
        cols.push(cc);
        const region = map.get(`${cr}:${cc}`);
        if (region) {
          clusterRegions.push(region);
          score += Number(region[key]);
        }
        for (const [dr, dc] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const nr = cr + dr;
          const nc = cc + dc;
          if (nr < 0 || nc < 0 || nr >= GRID || nc >= GRID || visited[nr][nc] || !mask[nr][nc]) continue;
          visited[nr][nc] = true;
          stack.push([nr, nc]);
        }
      }
      if (clusterRegions.length >= 2 || score > minScore * 2.4) {
        clusters.push({ rows, cols, score, regions: clusterRegions });
      }
    }
  }

  return clusters.sort((a, b) => b.score - a.score);
}

function clusterToBBox(cluster: Cluster): [number, number, number, number] {
  const minR = Math.min(...cluster.rows);
  const maxR = Math.max(...cluster.rows);
  const minC = Math.min(...cluster.cols);
  const maxC = Math.max(...cluster.cols);
  const pad = 0.35;
  const ymin = Math.round(clamp(((minR - pad) / GRID) * 1000, 20, 900));
  const xmin = Math.round(clamp(((minC - pad) / GRID) * 1000, 20, 900));
  const ymax = Math.round(clamp(((maxR + 1 + pad) / GRID) * 1000, ymin + 80, 980));
  const xmax = Math.round(clamp(((maxC + 1 + pad) / GRID) * 1000, xmin + 80, 980));
  return [ymin, xmin, ymax, xmax];
}

function confidenceFromScore(base: number, score: number, scale: number) {
  return Number(clamp(base + Math.min(0.16, score / scale), 0.62, 0.97).toFixed(2));
}

export function classifyPixels(
  pixels: Uint8ClampedArray,
  size: number,
  sourceName: string,
  timestampSeconds: number | null,
  frameIndex?: number,
): FrameObservation[] {
  const regions = regionStats(pixels, size);
  const vegClusters = clusterByScore(regions, 'veg', 22);
  const structClusters = clusterByScore(regions, 'struct', 48);
  const hydroClusters = clusterByScore(regions, 'hydro', 40);
  const roadClusters = clusterByScore(regions, 'road', 90);
  const obs: FrameObservation[] = [];
  const tsLabel = timestampSeconds === null ? 'still frame' : `${timestampSeconds.toFixed(2)}s`;

  const push = (
    cluster: Cluster,
    label: string,
    category: string,
    note: string,
    base: number,
    scale: number,
  ) => {
    const bbox = clusterToBBox(cluster);
    obs.push({
      label,
      category,
      confidence: confidenceFromScore(base, cluster.score, scale),
      evidenceNote: `${note} Detected at ${tsLabel} from spectral/texture clustering (score ${Math.round(cluster.score)}).`,
      timestampSeconds,
      boundingBox: bbox,
      frameIndex,
    });
  };

  if (structClusters[0]) {
    push(structClusters[0], 'Roof structure / Built envelope', 'structure', 'Rectilinear high-variance built form.', 0.84, 900);
  }
  if (vegClusters[0]) {
    push(vegClusters[0], 'Canopy foliage vegetation', 'vegetation', 'Positive Excess Green / NDVI-like canopy response.', 0.86, 700);
  }
  const elongatedRoad = roadClusters.find((cluster) => {
    const h = Math.max(...cluster.rows) - Math.min(...cluster.rows) + 1;
    const w = Math.max(...cluster.cols) - Math.min(...cluster.cols) + 1;
    return Math.max(w, h) / Math.max(1, Math.min(w, h)) >= 1.6;
  });
  if (elongatedRoad || roadClusters[0]) {
    push(elongatedRoad || roadClusters[0], 'Unpaved access corridor', 'road', 'Low-saturation linear ground reflectance.', 0.8, 800);
  }
  if (hydroClusters[0]) {
    push(hydroClusters[0], 'Surface water accumulation', 'hydrology', 'Dark / blue-biased low-reflectance surface.', 0.82, 650);
  }

  if (structClusters[0] && vegClusters[0]) {
    const sb = clusterToBBox(structClusters[0]);
    const nearEdge = sb[0] < 80 || sb[1] < 80 || sb[2] > 920 || sb[3] > 920;
    if (nearEdge) {
      obs.push({
        label: 'Boundary encroachment risk',
        category: 'encroachment',
        confidence: 0.78,
        evidenceNote: `Built envelope meets frame/parcel edge at ${tsLabel}. Survey verification required.`,
        timestampSeconds,
        boundingBox: [
          Math.min(sb[0], 40),
          Math.min(sb[1], 40),
          Math.max(sb[2], 200),
          Math.max(sb[3], 200),
        ],
        frameIndex,
      });
    }
  }

  if (obs.length === 0) {
    obs.push({
      label: 'Undifferentiated land cover',
      category: 'vegetation',
      confidence: 0.64,
      evidenceNote: `No high-confidence built/hydro signature at ${tsLabel}. Terrain treated as mixed land cover.`,
      timestampSeconds,
      boundingBox: [180, 180, 820, 820],
      frameIndex,
    });
  }

  return obs.map((item) => ({ ...item, evidenceNote: `${item.evidenceNote} Source: ${sourceName}.` }));
}

export async function classifyImageDataUrl(
  mediaData: string,
  sourceName: string,
  timestampSeconds: number | null,
  frameIndex?: number,
): Promise<FrameObservation[]> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = ANALYZE_SIZE;
      canvas.height = ANALYZE_SIZE;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        resolve([]);
        return;
      }
      ctx.drawImage(img, 0, 0, ANALYZE_SIZE, ANALYZE_SIZE);
      const data = ctx.getImageData(0, 0, ANALYZE_SIZE, ANALYZE_SIZE);
      resolve(classifyPixels(data.data, ANALYZE_SIZE, sourceName, timestampSeconds, frameIndex));
    };
    img.onerror = () => resolve([]);
    img.src = mediaData.startsWith('data:') ? mediaData : `data:image/jpeg;base64,${mediaData}`;
  });
}

function captureFrame(
  video: HTMLVideoElement,
  drawW: number,
  drawH: number,
): { poster: string; pixels: ImageData } {
  const posterCanvas = document.createElement('canvas');
  posterCanvas.width = drawW;
  posterCanvas.height = drawH;
  const posterCtx = posterCanvas.getContext('2d');
  posterCtx?.drawImage(video, 0, 0, drawW, drawH);

  const analysisCanvas = document.createElement('canvas');
  analysisCanvas.width = ANALYZE_SIZE;
  analysisCanvas.height = ANALYZE_SIZE;
  const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });
  analysisCtx?.drawImage(video, 0, 0, ANALYZE_SIZE, ANALYZE_SIZE);
  const pixels = analysisCtx?.getImageData(0, 0, ANALYZE_SIZE, ANALYZE_SIZE);

  return {
    poster: posterCanvas.toDataURL('image/jpeg', 0.72),
    pixels: pixels || new ImageData(ANALYZE_SIZE, ANALYZE_SIZE),
  };
}

export async function extractAndAnalyzeVideo(
  file: File,
  sourceName: string,
  onProgress?: (status: string) => void,
): Promise<{
  frames: ExtractedFrame[];
  durationSeconds: number;
  observations: FrameObservation[];
  posterData: string;
  sampledForModel: ExtractedFrame[];
}> {
  onProgress?.('Decoding full video timeline…');
  const video = await loadVideo(file);
  const objectUrl = video.src;
  const durationSeconds = video.duration;
  const timestamps = buildTimestamps(durationSeconds);
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const drawW = vw >= vh ? 640 : Math.round((640 * vw) / vh);
  const drawH = vw >= vh ? Math.round((640 * vh) / vw) : 640;

  const frames: ExtractedFrame[] = [];
  const observations: FrameObservation[] = [];

  try {
    for (let i = 0; i < timestamps.length; i++) {
      const ts = timestamps[i];
      await seekVideo(video, ts);
      const captured = captureFrame(video, drawW, drawH);
      frames.push({
        data: captured.poster,
        mimeType: 'image/jpeg',
        timestampSeconds: ts,
        frameIndex: i,
      });
      observations.push(
        ...classifyPixels(captured.pixels.data, ANALYZE_SIZE, sourceName, ts, i),
      );
      if (i === 0 || i % 12 === 0 || i === timestamps.length - 1) {
        onProgress?.(`Analyzing frame ${i + 1} / ${timestamps.length} (${ts.toFixed(1)}s of ${durationSeconds.toFixed(1)}s)`);
      }
    }
  } finally {
    URL.revokeObjectURL(objectUrl);
    video.removeAttribute('src');
    video.load();
  }

  const modelStride = Math.max(1, Math.ceil(frames.length / 16));
  const sampledForModel = frames.filter((_, idx) => idx % modelStride === 0).slice(0, 16);

  return {
    frames,
    durationSeconds,
    observations,
    posterData: frames[0]?.data || '',
    sampledForModel,
  };
}

export function mergeTemporalObservations(raw: FrameObservation[]): FrameObservation[] {
  const sorted = [...raw].sort((a, b) => (a.timestampSeconds ?? 0) - (b.timestampSeconds ?? 0));
  const tracks: Array<FrameObservation & { count: number; lastTs: number }> = [];

  const iou = (a: [number, number, number, number], b: [number, number, number, number]) => {
    const y1 = Math.max(a[0], b[0]);
    const x1 = Math.max(a[1], b[1]);
    const y2 = Math.min(a[2], b[2]);
    const x2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, y2 - y1) * Math.max(0, x2 - x1);
    const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
    const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / (areaA + areaB - inter);
  };

  for (const obs of sorted) {
    const ts = obs.timestampSeconds ?? 0;
    const match = tracks.find(
      (track) =>
        track.category === obs.category &&
        track.label === obs.label &&
        ts - track.lastTs <= 1.05 &&
        iou(track.boundingBox, obs.boundingBox) > 0.28,
    );
    if (!match) {
      tracks.push({ ...obs, count: 1, lastTs: ts });
      continue;
    }
    match.count += 1;
    match.lastTs = ts;
    if (obs.confidence > match.confidence) {
      match.confidence = obs.confidence;
      match.boundingBox = obs.boundingBox;
      match.timestampSeconds = obs.timestampSeconds;
      match.frameIndex = obs.frameIndex;
      match.evidenceNote = obs.evidenceNote;
    }
  }

  return tracks.map((track) => ({
    label: track.label,
    category: track.category,
    confidence: Number(clamp(track.confidence + Math.min(0.06, track.count / 80), 0.62, 0.98).toFixed(2)),
    evidenceNote: `${track.evidenceNote} Temporal track across ${track.count} analyzed frame${track.count === 1 ? '' : 's'}.`,
    timestampSeconds: track.timestampSeconds,
    boundingBox: track.boundingBox,
    frameIndex: track.frameIndex,
  }));
}

export function sampleFramesForApi(frames: ExtractedFrame[], limit = 16): ExtractedFrame[] {
  if (frames.length <= limit) return frames;
  const out: ExtractedFrame[] = [];
  for (let i = 0; i < limit; i++) {
    const idx = Math.round((i / (limit - 1)) * (frames.length - 1));
    out.push(frames[idx]);
  }
  return out;
}
