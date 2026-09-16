import type { AnalysisDetail, Observation } from "@workspace/api-zod";

export interface MultimodalRequest {
  title: string;
  sourceType: "image" | "video";
  sourceName: string;
  hasGeospatialMetadata: boolean;
  captureDate: string | null;
  mediaData: string;
  mediaMimeType: string;
  frames?: Array<{
    data: string;
    mimeType: string;
    timestampSeconds: number;
    frameIndex?: number;
  }>;
  durationSeconds?: number;
  analyzedFrameCount?: number;
}

interface RawObservation {
  label: string;
  category: string;
  confidence: number;
  evidenceNote: string;
  timestampSeconds?: number | null;
  boundingBox?: [number, number, number, number];
}

/**
 * Runs Gemini VLM analysis if GEMINI_API_KEY / GOOGLE_API_KEY is available
 * in env. Otherwise returns an honest "not configured" result -- there is
 * no real local vision engine to fall back to (see noProviderConfigured()).
 */
export async function runMultimodalAnalysis(
  input: MultimodalRequest,
): Promise<Omit<AnalysisDetail, "id" | "createdAt" | "verifiedObservationCount">> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;

  if (apiKey) {
    try {
      const geminiResult = await callGeminiVLM(input, apiKey);
      if (geminiResult) {
        return geminiResult;
      }
    } catch (err) {
      console.warn("Gemini VLM API call failed, returning an honest not-configured result:", err);
    }
  }

  return noProviderConfigured(input);
}

function stripBase64(data: string): string {
  return data.replace(/^data:[^;]+;base64,/, "");
}

function systemPrompt(input: MultimodalRequest): string {
  const duration = input.durationSeconds ? `${input.durationSeconds.toFixed(2)}s` : "unknown duration";
  const frameHint =
    input.sourceType === "video"
      ? `This is a drone/flight VIDEO covering the FULL timeline (${duration}). Each attached image is a timestamped frame from that video. Produce observations for EVERY provided frame. Use the exact timestampSeconds given next to each frame. Do not collapse the video into a single still.`
      : "This is a single aerial still / orthomosaic.";

  return `You are Geo-VLM Sentinel, an expert aerial GIS and remote-sensing analyst.
${frameHint}

Rules:
- Identify only features you can actually see: structures/roofs, roads/paths, vegetation/canopy, water, parcel boundaries, encroachment.
- category MUST be one of: structure, road, agriculture, hydrology, vegetation, boundary, encroachment
- boundingBox is [ymin, xmin, ymax, xmax] in 0-1000 normalized image coordinates. Boxes must tightly fit the visible object.
- confidence is calibrated: 0.70-0.84 uncertain, 0.85-0.93 clear, 0.94+ unmistakable. Never invent high confidence.
- Do not report a class unless visual evidence is present.
- For video, include timestampSeconds matching the source frame.

Return strictly valid JSON with no markdown wrapping:
{
  "modelLabel": "Gemini 2.5 Flash VLM • Full-timeline Aerial Engine",
  "observations": [
    {
      "label": "Short feature name",
      "category": "structure",
      "confidence": 0.88,
      "evidenceNote": "What is visible and where.",
      "timestampSeconds": 0,
      "boundingBox": [120, 200, 410, 560]
    }
  ]
}`;
}

async function generateFromParts(
  url: string,
  prompt: string,
  parts: Array<Record<string, unknown>>,
): Promise<RawObservation[] | null> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }, ...parts] }],
      generationConfig: {
        temperature: 0.1,
        topP: 0.8,
        responseMimeType: "application/json",
      },
    }),
  });

  if (!response.ok) {
    console.error("Gemini API HTTP Error:", response.status, await response.text());
    return null;
  }

  const json = (await response.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
  };
  const textContent = json?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!textContent) return null;

  const parsed = JSON.parse(textContent) as { observations?: RawObservation[]; modelLabel?: string };
  return parsed.observations || [];
}

async function callGeminiVLM(
  input: MultimodalRequest,
  apiKey: string,
): Promise<Omit<AnalysisDetail, "id" | "createdAt" | "verifiedObservationCount"> | null> {
  const modelName = "gemini-2.5-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;
  const frames = input.frames && input.frames.length > 0 ? input.frames : [];
  const prompt = systemPrompt(input);

  const batches: RawObservation[] = [];

  if (input.sourceType === "video" && frames.length > 0) {
    const batchSize = 6;
    for (let i = 0; i < frames.length; i += batchSize) {
      const slice = frames.slice(i, i + batchSize);
      const parts: Array<Record<string, unknown>> = [];
      for (const frame of slice) {
        parts.push({
          text: `Frame ${frame.frameIndex ?? i} at t=${frame.timestampSeconds.toFixed(2)}s of the source video. Analyze this exact frame.`,
        });
        parts.push({
          inlineData: {
            mimeType: frame.mimeType || "image/jpeg",
            data: stripBase64(frame.data),
          },
        });
      }
      const obs = await generateFromParts(url, prompt, parts);
      if (obs) {
        batches.push(
          ...obs.map((item, idx) => ({
            ...item,
            timestampSeconds:
              item.timestampSeconds ?? slice[Math.min(idx, slice.length - 1)]?.timestampSeconds ?? null,
          })),
        );
      }
    }
  } else {
    const parts: Array<Record<string, unknown>> = [];
    if (input.mediaData) {
      parts.push({
        inlineData: {
          mimeType: input.mediaMimeType || "image/jpeg",
          data: stripBase64(input.mediaData),
        },
      });
    }
    const obs = await generateFromParts(url, prompt, parts);
    if (obs) batches.push(...obs);
  }

  if (batches.length === 0) return null;

  const observations: Observation[] = batches.map((obs, idx) => ({
    id: `obs-gemini-${Date.now()}-${idx + 1}`,
    label: obs.label || "Detected aerial feature",
    category: obs.category || "vegetation",
    confidence: Math.min(0.98, Math.max(0.55, obs.confidence || 0.78)),
    reviewStatus: "unreviewed",
    evidenceNote: obs.evidenceNote || "VLM model detection from aerial frame.",
    timestampSeconds:
      obs.timestampSeconds ??
      (input.sourceType === "video" && frames[idx]
        ? frames[idx].timestampSeconds
        : input.sourceType === "video"
          ? Number(((idx / Math.max(1, batches.length - 1)) * (input.durationSeconds || 0)).toFixed(2))
          : null),
    sourceMedia: input.sourceName,
    boundingBox:
      obs.boundingBox && obs.boundingBox.length === 4
        ? obs.boundingBox
        : undefined,
  }));

  const hasHighRisk = observations.some((o) => o.category === "encroachment" || o.category === "boundary");

  return {
    title: input.title,
    sourceType: input.sourceType,
    sourceName: input.sourceName,
    status: hasHighRisk ? "needs_attention" : "needs_review",
    observationCount: observations.length,
    hasGeospatialMetadata: input.hasGeospatialMetadata,
    captureDate: input.captureDate || new Date().toISOString().slice(0, 10),
    modelLabel: "Gemini 2.5 Flash VLM • Full-timeline Aerial Engine",
    observations,
    mediaData: input.mediaData,
  };
}

/**
 * No API key configured -- there is no real local vision engine, and this
 * project's own stance (see geocadastra/api/advisory.py's ward_brief()) is
 * that a missing capability returns an honest "not configured" state, not
 * a plausible-looking fabricated result. The function this replaced
 * (`generateDynamicCVAnalysis`) never looked at the uploaded image at
 * all -- it cycled through 4 fixed labels by array index and computed
 * "bounding boxes" from arithmetic on a loop counter, under a comment
 * that called it "an intelligent Computer Vision feature extraction
 * algorithm." Zero observations and a status that routes to human review
 * is the honest answer until a real fallback model is wired in.
 */
function noProviderConfigured(
  input: MultimodalRequest,
): Omit<AnalysisDetail, "id" | "createdAt" | "verifiedObservationCount"> {
  return {
    title: input.title,
    sourceType: input.sourceType,
    sourceName: input.sourceName,
    status: "needs_review",
    observationCount: 0,
    hasGeospatialMetadata: input.hasGeospatialMetadata,
    captureDate: input.captureDate || new Date().toISOString().slice(0, 10),
    modelLabel: "Not analyzed -- no GEMINI_API_KEY/GOOGLE_API_KEY configured on the server",
    observations: [],
    mediaData: input.mediaData,
  };
}
