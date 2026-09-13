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

type Parcel = ReturnType<typeof GetParcelResponse.parse>;
type ProcessingRun = ReturnType<typeof ListProcessingRunsResponse.parse>[number];

const regions = [
  {
    id: "ward-42",
    name: "Ward 42 · Delhi Urban Zone",
    type: "Municipal ward",
    processedSqKm: 18.42,
    parcelCount: 1284,
    updatedAt: "13 Sep 2026 · 09:42 IST",
  },
  {
    id: "sector-18",
    name: "Sector 18 · Noida",
    type: "Planned sector",
    processedSqKm: 12.86,
    parcelCount: 916,
    updatedAt: "12 Sep 2026 · 17:10 IST",
  },
  {
    id: "ranchi-east",
    name: "Ranchi East · Jharkhand",
    type: "Urban circle",
    processedSqKm: 9.72,
    parcelCount: 604,
    updatedAt: "11 Sep 2026 · 14:25 IST",
  },
];

const parcels: Parcel[] = [
  {
    id: "parcel-0421",
    ulpin: "IN-DL-42-001284",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 842.6,
    ownership: "Verified · Private",
    confidence: 0.982,
    status: "verified",
    geometry: [
      [18, 28],
      [28, 23],
      [39, 31],
      [35, 44],
      [22, 48],
      [18, 28],
    ],
    updatedAt: "13 Sep 2026 · 09:38 IST",
  },
  {
    id: "parcel-0422",
    ulpin: "IN-DL-42-001285",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 618.2,
    ownership: "Pending verification",
    confidence: 0.914,
    status: "review",
    geometry: [
      [39, 31],
      [50, 25],
      [61, 35],
      [58, 50],
      [35, 44],
      [39, 31],
    ],
    updatedAt: "13 Sep 2026 · 09:36 IST",
  },
  {
    id: "parcel-0423",
    ulpin: "IN-DL-42-001286",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 1106.4,
    ownership: "Verified · Institutional",
    confidence: 0.967,
    status: "verified",
    geometry: [
      [61, 35],
      [75, 29],
      [84, 41],
      [78, 59],
      [58, 50],
      [61, 35],
    ],
    updatedAt: "13 Sep 2026 · 09:34 IST",
  },
  {
    id: "parcel-0424",
    ulpin: "IN-DL-42-001287",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 452.8,
    ownership: "Flagged · Encroachment",
    confidence: 0.889,
    status: "flagged",
    geometry: [
      [22, 48],
      [35, 44],
      [46, 63],
      [40, 76],
      [26, 71],
      [22, 48],
    ],
    updatedAt: "13 Sep 2026 · 09:29 IST",
  },
  {
    id: "parcel-0425",
    ulpin: "IN-DL-42-001288",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 734.1,
    ownership: "Verified · Private",
    confidence: 0.951,
    status: "verified",
    geometry: [
      [46, 63],
      [58, 50],
      [78, 59],
      [69, 78],
      [54, 83],
      [40, 76],
      [46, 63],
    ],
    updatedAt: "13 Sep 2026 · 09:27 IST",
  },
  {
    id: "parcel-0426",
    ulpin: "IN-DL-42-001289",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    areaSqM: 526.7,
    ownership: "Pending verification",
    confidence: 0.923,
    status: "review",
    geometry: [
      [78, 59],
      [91, 55],
      [94, 70],
      [82, 86],
      [69, 78],
      [78, 59],
    ],
    updatedAt: "13 Sep 2026 · 09:22 IST",
  },
];

const processingRuns: ProcessingRun[] = [
  {
    id: "run-0913",
    name: "Ward 42 · Q3 baseline extraction",
    regionId: "ward-42",
    regionName: "Ward 42 · Delhi Urban Zone",
    status: "completed",
    progress: 100,
    currentStep: "Topology graph generated",
    steps: [
      "Raster tiling & COG streaming",
      "DSM/DTM fusion segmentation",
      "Building & road extraction",
      "Topology graph generation",
    ],
    startedAt: "13 Sep 2026 · 07:15 IST",
  },
];

const changes = [
  {
    id: "change-01",
    title: "Boundary expansion detected",
    regionName: "Ward 42 · Delhi Urban Zone",
    category: "Built-up expansion",
    areaSqM: 184.3,
    detectedAt: "13 Sep 2026 · 08:52 IST",
    severity: "high",
  },
  {
    id: "change-02",
    title: "Road right-of-way variance",
    regionName: "Sector 18 · Noida",
    category: "Encroachment",
    areaSqM: 72.8,
    detectedAt: "12 Sep 2026 · 16:19 IST",
    severity: "medium",
  },
  {
    id: "change-03",
    title: "New rooftop structure",
    regionName: "Ward 42 · Delhi Urban Zone",
    category: "Vertical change",
    areaSqM: 46.1,
    detectedAt: "12 Sep 2026 · 11:04 IST",
    severity: "low",
  },
];

let topology = {
  scannedAt: "13 Sep 2026 · 09:40 IST",
  errorCount: 12,
  fixedCount: 0,
  overlapCount: 5,
  sliverCount: 4,
  ringCount: 3,
  status: "attention",
};

function dashboard() {
  const activeRun =
    processingRuns.find((run) => run.status === "processing") ?? null;
  return GetDashboardResponse.parse({
    areaProcessedSqKm: 41.0,
    parcelsExtracted: 2804,
    topologyErrors: topology.errorCount,
    encroachments: changes.filter((change) => change.severity === "high").length,
    accuracyScore: 94.8,
    activeRun,
    recentActivity: [
      {
        id: "activity-01",
        title: "Topology scan completed",
        detail: "12 boundary anomalies require review in Ward 42",
        time: "8 min ago",
        tone: "warning",
      },
      {
        id: "activity-02",
        title: "Q3 imagery harmonized",
        detail: "ORI, DSM and DTM aligned for 18.42 sq. km",
        time: "32 min ago",
        tone: "success",
      },
      {
        id: "activity-03",
        title: "Field sync received",
        detail: "24 GNSS observations from survey team DEL-07",
        time: "1 hr ago",
        tone: "info",
      },
      {
        id: "activity-04",
        title: "Encroachment flagged",
        detail: "184.3 sq. m variance detected on ULPIN IN-DL-42-001287",
        time: "2 hrs ago",
        tone: "danger",
      },
    ],
  });
}

router.get("/dashboard", (_req, res) => {
  res.json(dashboard());
});

router.get("/regions", (_req, res) => {
  res.json(ListRegionsResponse.parse(regions));
});

router.get("/parcels", (req, res) => {
  const query = ListParcelsQueryParams.parse(req.query);
  const result = parcels.filter(
    (parcel) =>
      (!query.regionId || parcel.regionId === query.regionId) &&
      (!query.status || parcel.status === query.status),
  );
  res.json(ListParcelsResponse.parse(result));
});

router.get("/parcels/:id", (req, res) => {
  const { id } = GetParcelParams.parse(req.params);
  const parcel = parcels.find((item) => item.id === id);
  if (!parcel) {
    res.status(404).json({ error: "Parcel not found" });
    return;
  }
  res.json(GetParcelResponse.parse(parcel));
});

router.patch("/parcels/:id", (req, res) => {
  const { id } = UpdateParcelParams.parse(req.params);
  const body = UpdateParcelBody.parse(req.body);
  const parcel = parcels.find((item) => item.id === id);
  if (!parcel) {
    res.status(404).json({ error: "Parcel not found" });
    return;
  }
  if (body.status) parcel.status = body.status;
  if (body.geometry) parcel.geometry = body.geometry;
  parcel.updatedAt = "13 Sep 2026 · just now";
  res.json(UpdateParcelResponse.parse(parcel));
});

router.get("/processing/runs", (_req, res) => {
  res.json(ListProcessingRunsResponse.parse(processingRuns));
});

router.post("/processing/runs", (req, res) => {
  const body = CreateProcessingRunBody.parse(req.body);
  const region = regions.find((item) => item.id === body.regionId) ?? regions[0];
  const run: ProcessingRun = {
    id: `run-${Date.now()}`,
    name: `${region.name} · ${body.dataset} inference`,
    regionId: region.id,
    regionName: region.name,
    status: "processing",
    progress: 14,
    currentStep: "Raster tiling & COG streaming",
    steps: [
      "Raster tiling & COG streaming",
      "DSM/DTM fusion segmentation",
      "Building & road extraction",
      "Topology graph generation",
    ],
    startedAt: "13 Sep 2026 · just now",
  };
  processingRuns.unshift(run);
  res.status(201).json(run);
});

router.post("/topology/scan", (_req, res) => {
  topology = {
    ...topology,
    scannedAt: "13 Sep 2026 · just now",
    status: "attention",
  };
  res.json(ScanTopologyResponse.parse(topology));
});

router.post("/topology/fix", (_req, res) => {
  topology = {
    ...topology,
    scannedAt: "13 Sep 2026 · just now",
    errorCount: 0,
    fixedCount: 12,
    overlapCount: 0,
    sliverCount: 0,
    ringCount: 0,
    status: "healthy",
  };
  res.json(FixTopologyResponse.parse(topology));
});

router.get("/changes", (_req, res) => {
  res.json(ListChangesResponse.parse(changes));
});

router.post("/exports", (req, res) => {
  const body = CreateExportBody.parse(req.body);
  res.status(201).json({
    id: `export-${Date.now()}`,
    format: body.format,
    regionId: body.regionId,
    status: "ready",
    createdAt: "13 Sep 2026 · just now",
    fileName: `bhoomiai-${body.regionId}-${body.format.toLowerCase().replaceAll(" ", "-")}.zip`,
  });
});

export default router;