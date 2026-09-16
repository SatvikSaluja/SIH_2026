import { Router, type IRouter } from "express";
import {
  CreateAnalysisBody,
  GetAnalysisParams,
  GetAnalysisReportParams,
  GetSentinelOverviewResponse,
  GetAnalysisResponse,
  GetAnalysisReportResponse,
  ListAnalysesResponse,
  VerifyAnalysisBody,
  VerifyAnalysisParams,
  type AnalysisDetail,
  type Observation,
} from "@workspace/api-zod";
import { store } from "../lib/store.js";
import { runMultimodalAnalysis, type MultimodalRequest } from "../lib/vlm.js";

const router: IRouter = Router();

// Overview summary
router.get("/sentinel/overview", (_req, res) => {
  const overview = store.getOverview();
  res.json(GetSentinelOverviewResponse.parse(overview));
});

// List recent analyses
router.get("/sentinel/analyses", (_req, res) => {
  const data = store.listAnalyses();
  res.json(ListAnalysesResponse.parse(data));
});

// Save direct analysis input record
router.post("/sentinel/analyses", (req, res) => {
  const input = CreateAnalysisBody.parse(req.body);
  const id = `analysis-${Date.now()}`;
  
  const observations: Observation[] = (input.observations ?? []).map(
    (obs, idx) => ({
      id: `${id}-obs-${idx + 1}`,
      label: obs.label,
      category: obs.category,
      confidence: obs.confidence,
      reviewStatus: obs.reviewStatus,
      evidenceNote: obs.evidenceNote,
      timestampSeconds: obs.timestampSeconds ?? null,
      sourceMedia: obs.sourceMedia,
      boundingBox: [
        150 + ((idx * 150) % 500),
        100 + ((idx * 180) % 550),
        400 + ((idx * 150) % 500),
        350 + ((idx * 180) % 550),
      ],
    }),
  );

  const detail: AnalysisDetail = {
    id,
    title: input.title,
    sourceType: input.sourceType,
    sourceName: input.sourceName,
    status: input.status,
    createdAt: new Date().toISOString(),
    observationCount: observations.length,
    verifiedObservationCount: observations.filter(
      (o) => o.reviewStatus === "confirmed",
    ).length,
    hasGeospatialMetadata: input.hasGeospatialMetadata,
    captureDate: input.captureDate || new Date().toISOString().slice(0, 10),
    modelLabel: input.modelLabel || "Geo-VLM Sentinel Engine",
    observations,
  };

  const summary = store.saveAnalysis(detail);
  res.status(201).json(summary);
});

// Multimodal VLM media analysis (Base64 image / frame extraction)
router.post("/sentinel/analyze-media", async (req, res) => {
  try {
    const payload: MultimodalRequest = req.body;
    if (!payload.mediaData && (!payload.frames || payload.frames.length === 0)) {
      res.status(400).json({ error: "Missing mediaData or frame inputs for multimodal analysis" });
      return;
    }

    const result = await runMultimodalAnalysis(payload);
    const id = `analysis-vlm-${Date.now()}`;
    const detail: AnalysisDetail = {
      id,
      ...result,
      createdAt: new Date().toISOString(),
      verifiedObservationCount: result.observations.filter(
        (o) => o.reviewStatus === "confirmed",
      ).length,
    };

    store.saveAnalysis(detail);
    res.status(201).json(detail);
  } catch (err) {
    console.error("Error running multimodal media analysis:", err);
    res.status(500).json({ error: "Multimodal AI inference failed" });
  }
});

// Get detailed analysis record with observations
router.get("/sentinel/analyses/:analysisId", (req, res) => {
  const { analysisId } = GetAnalysisParams.parse(req.params);
  const detail = store.getAnalysis(analysisId);
  if (!detail) {
    res.status(404).json({ error: "Analysis not found" });
    return;
  }
  res.json(detail);
});

// Update verification decision or observation statuses
router.patch("/sentinel/analyses/:analysisId", (req, res) => {
  const { analysisId } = VerifyAnalysisParams.parse(req.params);
  const input = VerifyAnalysisBody.parse(req.body);
  const observationUpdates = req.body.observations;

  const summary = store.verifyAnalysis(analysisId, input.status, observationUpdates);
  if (!summary) {
    res.status(404).json({ error: "Analysis not found" });
    return;
  }
  res.json(summary);
});

// Get preliminary evidence report
router.get("/sentinel/analyses/:analysisId/report", (req, res) => {
  const { analysisId } = GetAnalysisReportParams.parse(req.params);
  const report = store.getReport(analysisId);
  if (!report) {
    res.status(404).json({ error: "Analysis not found" });
    return;
  }
  res.json(GetAnalysisReportResponse.parse(report));
});

// Live Drone Telemetry endpoint
router.get("/sentinel/telemetry", (_req, res) => {
  const now = Date.now();
  const cycle = (now / 1000) % 360;
  
  // Sine/cosine physics flight dynamics simulation
  const altitude = Number((120 + Math.sin(cycle * 0.05) * 15).toFixed(1));
  const speed = Number((8.5 + Math.cos(cycle * 0.08) * 2.2).toFixed(1));
  const heading = Math.round((cycle * 2) % 360);
  const battery = Math.max(15, Math.round(98 - ((now / 60000) % 75)));
  const pitch = Number((Math.sin(cycle * 0.1) * 4.5).toFixed(1));
  const roll = Number((Math.cos(cycle * 0.1) * 3.2).toFixed(1));
  const lat = Number((20.4625 + Math.sin(cycle * 0.02) * 0.003).toFixed(5));
  const lng = Number((85.8792 + Math.cos(cycle * 0.02) * 0.003).toFixed(5));

  res.json({
    timestamp: new Date().toISOString(),
    droneId: "SENTINEL-DRONE-X4",
    status: "FLIGHT_ACTIVE",
    telemetry: {
      altitudeMeters: altitude,
      airspeedMps: speed,
      headingDegrees: heading,
      batteryPercentage: battery,
      pitchDegrees: pitch,
      rollDegrees: roll,
      latitude: lat,
      longitude: lng,
      signalStrength: 96,
      satellitesConnected: 18,
      gimbalAngle: -45,
    },
  });
});

export default router;