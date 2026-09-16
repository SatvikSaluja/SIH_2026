import fs from "node:fs";
import path from "node:path";
import type { Analysis, AnalysisDetail, AnalysisReport } from "@workspace/api-zod";

const DATA_DIR = path.resolve(process.cwd(), "data");
const STORE_FILE = path.join(DATA_DIR, "sentinel_ledger.json");

const disclaimer =
  "Preliminary evidence only. This output does not determine ownership, legal boundaries, or official cadastral status. Authorized survey verification is required.";

const initialSeedData: AnalysisDetail[] = [
  {
    id: "analysis-demo-001",
    title: "Northfield • preliminary survey review",
    sourceType: "image",
    sourceName: "northfield-orthomosaic.jpg",
    status: "needs_attention",
    createdAt: "2026-09-16T09:24:00.000Z",
    observationCount: 3,
    verifiedObservationCount: 0,
    hasGeospatialMetadata: true,
    captureDate: "2026-09-16",
    modelLabel: "Geo-VLM Sentinel • Aerial Vision Engine",
    observations: [
      {
        id: "analysis-demo-001-obs-1",
        label: "Built structure",
        category: "structure",
        confidence: 0.87,
        reviewStatus: "unreviewed",
        evidenceNote:
          "Rectilinear roof-like form detected in the source frame. Height profile ~4.2m.",
        timestampSeconds: null,
        sourceMedia: "northfield-orthomosaic.jpg",
        boundingBox: [150, 200, 420, 520],
      },
      {
        id: "analysis-demo-001-obs-2",
        label: "Access corridor",
        category: "road",
        confidence: 0.74,
        reviewStatus: "unreviewed",
        evidenceNote:
          "Linear cleared corridor visible along western boundary edge.",
        timestampSeconds: null,
        sourceMedia: "northfield-orthomosaic.jpg",
        boundingBox: [480, 120, 850, 310],
      },
      {
        id: "analysis-demo-001-obs-3",
        label: "Unregistered boundary hedge",
        category: "boundary",
        confidence: 0.61,
        reviewStatus: "unreviewed",
        evidenceNote:
          "Dense hedge vegetation break along field perimeter; verify against survey notes.",
        timestampSeconds: null,
        sourceMedia: "northfield-orthomosaic.jpg",
        boundingBox: [220, 600, 680, 880],
      },
    ],
  },
  {
    id: "analysis-demo-002",
    title: "Kandhamal valley flight 03",
    sourceType: "video",
    sourceName: "KND-12_flight_03.mp4",
    status: "verified",
    createdAt: "2026-09-15T14:32:00.000Z",
    observationCount: 4,
    verifiedObservationCount: 4,
    hasGeospatialMetadata: true,
    captureDate: "2026-09-15",
    modelLabel: "Geo-VLM Sentinel • Flight Analyzer",
    observations: [
      {
        id: "analysis-demo-002-obs-1",
        label: "Standing water body",
        category: "hydrology",
        confidence: 0.94,
        reviewStatus: "confirmed",
        evidenceNote: "Irrigation channel overflow along terrace embankment.",
        timestampSeconds: 12,
        sourceMedia: "KND-12_flight_03.mp4",
        boundingBox: [300, 150, 580, 450],
      },
      {
        id: "analysis-demo-002-obs-2",
        label: "Paved access track",
        category: "road",
        confidence: 0.89,
        reviewStatus: "confirmed",
        evidenceNote: "Primary block entrance track connecting village road.",
        timestampSeconds: 28,
        sourceMedia: "KND-12_flight_03.mp4",
        boundingBox: [100, 500, 400, 900],
      },
      {
        id: "analysis-demo-002-obs-3",
        label: "Canopy crop health variance",
        category: "agriculture",
        confidence: 0.76,
        reviewStatus: "confirmed",
        evidenceNote: "Uniform greenness index across parcel 14-B.",
        timestampSeconds: 45,
        sourceMedia: "KND-12_flight_03.mp4",
        boundingBox: [600, 300, 900, 750],
      },
      {
        id: "analysis-demo-002-obs-4",
        label: "Pump house facility",
        category: "structure",
        confidence: 0.82,
        reviewStatus: "confirmed",
        evidenceNote: "Concrete pump house shed near river junction.",
        timestampSeconds: 64,
        sourceMedia: "KND-12_flight_03.mp4",
        boundingBox: [200, 250, 380, 420],
      },
    ],
  },
];

class SentinelStore {
  private cache = new Map<string, AnalysisDetail>();

  constructor() {
    this.init();
  }

  private init() {
    try {
      if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
      }
      if (fs.existsSync(STORE_FILE)) {
        const raw = fs.readFileSync(STORE_FILE, "utf-8");
        const list: AnalysisDetail[] = JSON.parse(raw);
        for (const item of list) {
          this.cache.set(item.id, item);
        }
      } else {
        for (const item of initialSeedData) {
          this.cache.set(item.id, item);
        }
        this.persist();
      }
    } catch (err) {
      console.error("Failed to load sentinel store from disk:", err);
      for (const item of initialSeedData) {
        this.cache.set(item.id, item);
      }
    }
  }

  private persist() {
    try {
      if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
      }
      const data = Array.from(this.cache.values());
      fs.writeFileSync(STORE_FILE, JSON.stringify(data, null, 2), "utf-8");
    } catch (err) {
      console.error("Failed to persist sentinel store:", err);
    }
  }

  private toSummary(analysis: AnalysisDetail): Analysis {
    return {
      id: analysis.id,
      title: analysis.title,
      sourceType: analysis.sourceType,
      sourceName: analysis.sourceName,
      status: analysis.status,
      createdAt: analysis.createdAt,
      observationCount: analysis.observations.length,
      verifiedObservationCount: analysis.observations.filter(
        (o) => o.reviewStatus === "confirmed",
      ).length,
      hasGeospatialMetadata: analysis.hasGeospatialMetadata,
      captureDate: analysis.captureDate,
      modelLabel: analysis.modelLabel,
    };
  }

  public getOverview() {
    const values = Array.from(this.cache.values());
    const latest = values.sort((a, b) =>
      b.createdAt.localeCompare(a.createdAt),
    )[0];

    return {
      totalAnalyses: values.length,
      pendingReview: values.filter((item) => item.status === "needs_review")
        .length,
      verified: values.filter((item) => item.status === "verified").length,
      needsAttention: values.filter(
        (item) => item.status === "needs_attention",
      ).length,
      latestAnalysis: latest ? this.toSummary(latest) : null,
    };
  }

  public listAnalyses(): Analysis[] {
    return Array.from(this.cache.values())
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .map((item) => this.toSummary(item));
  }

  public getAnalysis(id: string): AnalysisDetail | null {
    return this.cache.get(id) || null;
  }

  public saveAnalysis(detail: AnalysisDetail): Analysis {
    this.cache.set(detail.id, detail);
    this.persist();
    return this.toSummary(detail);
  }

  public verifyAnalysis(
    id: string,
    status: "needs_review" | "verified" | "needs_attention",
    observationUpdates?: Array<{ id: string; reviewStatus: "unreviewed" | "confirmed" | "dismissed" }>,
  ): Analysis | null {
    const item = this.cache.get(id);
    if (!item) return null;

    item.status = status;
    if (observationUpdates && observationUpdates.length > 0) {
      const updateMap = new Map(observationUpdates.map((u) => [u.id, u.reviewStatus]));
      item.observations = item.observations.map((obs) => {
        if (updateMap.has(obs.id)) {
          return { ...obs, reviewStatus: updateMap.get(obs.id)! };
        }
        return obs;
      });
    }

    item.verifiedObservationCount = item.observations.filter(
      (o) => o.reviewStatus === "confirmed",
    ).length;

    this.cache.set(id, item);
    this.persist();
    return this.toSummary(item);
  }

  public getReport(id: string): AnalysisReport | null {
    const detail = this.cache.get(id);
    if (!detail) return null;
    return {
      generatedAt: new Date().toISOString(),
      title: `Preliminary evidence report • ${detail.title}`,
      disclaimer,
      analysis: detail,
    };
  }
}

export const store = new SentinelStore();
