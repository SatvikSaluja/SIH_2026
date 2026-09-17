import { Router, type IRouter } from 'express';
import { z } from 'zod';

const router: IRouter = Router();
const base = process.env.GEOCADASTRA_API_BASE ?? 'http://127.0.0.1:8000';
class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}
async function gc(path: string, init?: RequestInit) {
  const response = await fetch(`${base}${path}`, { ...init, signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new ApiError(response.status, await response.text());
  return response.json();
}
const jsonPost = (body?: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
interface Ward {
  ward_job_id: number; source: string; status: string; n_blocks: number; n_parcels: number;
  completed_blocks: number; area_sqkm: number; created_at: string; crs: string; synthetic: boolean;
}
const rid = (w: Ward) => `ward-${w.ward_job_id}`;
const name = (w: Ward) => `Ward ${w.ward_job_id} (${w.source})`;
const progress = (w: Ward) => w.n_blocks ? Math.floor(100 * w.completed_blocks / w.n_blocks) : 0;
async function wards(): Promise<Ward[]> { return (await gc('/wards')).wards; }
async function ward(id: unknown) {
  if (typeof id !== 'string' || !/^ward-[1-9]\d*$/.test(id)) throw new ApiError(400, 'Select a ward ID from the region list');
  const result = (await wards()).find(w => rid(w) === id);
  if (!result) throw new ApiError(404, 'Ward not found');
  return result;
}
function run(w: Ward) {
  return { id: `run-${w.ward_job_id}`, name: name(w), regionId: rid(w), regionName: name(w), status: w.status,
    progress: progress(w), currentStep: w.status, steps: ['ingest', 'process'], startedAt: w.created_at };
}
interface MapFeature {
  id: string; geometry: { type: 'Polygon'; coordinates: number[][][] };
  properties: { face_id: number; parcel_id: number | null; block_id: number; area_m2: number; updated_at: string };
}
async function geometry(w: Ward) {
  const features: MapFeature[] = [];
  let offset: number | null = 0;
  let coordinateSystem = '';
  do {
    const page = await gc(`/wards/${w.ward_job_id}/geometry?offset=${offset}&limit=2000`);
    features.push(...page.features);
    coordinateSystem = page.coordinate_system;
    const next = page.next_offset;
    if (next !== null && (!Number.isInteger(next) || next <= offset)) throw new ApiError(502, 'Invalid parcel pagination');
    offset = next;
  } while (offset !== null);
  return { features, coordinateSystem };
}
async function parcels(w: Ward) {
  const data = await geometry(w);
  return data.features.map(f => ({ id: String(f.properties.face_id), ulpin: `GC-${f.properties.parcel_id ?? f.properties.face_id}`,
    regionId: rid(w), regionName: name(w), areaSqM: f.properties.area_m2, ownership: null, confidence: null,
    status: f.properties.parcel_id === null ? 'unmatched' : 'matched', geometry: f.geometry,
    coordinateSystem: data.coordinateSystem, updatedAt: f.properties.updated_at }));
}
router.get('/dashboard', async (_req, res, next) => {
  try {
    const list = await wards();
    res.json({ areaProcessedSqKm: list.filter(w => w.status === 'done').reduce((s,w) => s + w.area_sqkm,0),
      parcelsExtracted: list.reduce((s,w) => s + w.n_parcels,0), topologyErrors: null, encroachments: null, accuracyScore: null,
      activeRun: list.some(w => w.status === 'running') ? run(list.find(w => w.status === 'running')!) : null,
      recentActivity: [...list].sort((a,b) => b.created_at.localeCompare(a.created_at)).slice(0,6).map(w => ({
        id: rid(w), title: `${name(w)} ingested`, detail: `Current status: ${w.status}`, time: w.created_at, tone: 'info' })) });
  } catch (e) { next(e); }
});
router.get('/regions', async (_req,res,next) => {
  try { res.json((await wards()).map(w => ({id:rid(w), name:name(w), type:w.synthetic?'Synthetic ward':'Georeferenced ward', areaSqKm:w.area_sqkm, status:w.status}))); }
  catch(e) { next(e); }
});
router.get('/parcels', async (req,res,next) => {
  try {
    const list = req.query.regionId ? [await ward(req.query.regionId)] : await wards();
    const out = (await Promise.all(list.map(parcels))).flat();
    res.json(req.query.status ? out.filter(p => p.status === req.query.status) : out);
  } catch(e) { next(e); }
});
router.get('/parcels/:id', async (req,res,next) => {
  try {
    for (const w of await wards()) {
      const found = (await parcels(w)).find(p => p.id === req.params.id);
      if(found) { res.json(found); return; }
    }
    throw new ApiError(404,'Parcel not found');
  } catch(e) { next(e); }
});
router.patch('/parcels/:id', (_req,res) => { res.status(501).json({error:'Parcel status editing is unavailable. Use the recorded ward node-edit workflow.'}); });
router.get('/processing/runs', async (_req,res,next) => {
  try { res.json((await wards()).map(run)); } catch(e) { next(e); }
});
router.post('/processing/runs', async (req,res,next) => {
  try {
    // Existing wards already identify their input data. Never manufacture a new dataset on Run.
    const input = z.object({regionId:z.string(), dataset:z.string().optional()}).strict().parse(req.body);
    const selected = await ward(input.regionId);
    if (input.dataset !== undefined && input.dataset !== selected.source) throw new ApiError(409,'Dataset does not match the selected ward source');
    await gc(`/wards/${selected.ward_job_id}/run`, jsonPost());
    res.status(202).json(run(await ward(input.regionId)));
  } catch(e) { next(e); }
});
router.post('/processing/synthetic', async (req,res,next) => {
  try {
    const body = z.object({seed:z.number().int(), width:z.number().positive().max(500).default(50), height:z.number().positive().max(500).default(35)}).strict().parse(req.body);
    res.status(201).json(await gc('/wards/ingest',jsonPost({...body,n_arterial_h:1,n_arterial_v:0})));
  } catch(e) { next(e); }
});
router.post('/topology/scan', async (req,res,next) => {
  try { const w = await ward(req.body?.regionId); res.json(await gc(`/wards/${w.ward_job_id}/topology`)); } catch(e) { next(e); }
});
router.post('/topology/fix', (_req,res) => { res.status(501).json({error:'Automatic repair is unavailable. Inspect the detected faces and apply a recorded node edit.'}); });
router.get('/changes', (_req,res) => { res.json([]); });
router.post('/exports', async(req,res,next) => {
  try {
    const body = z.object({regionId:z.string(),format:z.literal('GeoJSON')}).parse(req.body);
    const w = await ward(body.regionId); const data = await geometry(w);
    if(data.coordinateSystem !== 'EPSG:4326') throw new ApiError(422,'Synthetic local coordinates cannot be exported as geographic GeoJSON');
    res.status(201).json({id:`export-${Date.now()}`,format:'GeoJSON',regionId:rid(w),status:'ready',createdAt:new Date().toISOString(),fileName:`${rid(w)}.geojson`,geojson:{type:'FeatureCollection',features:data.features}});
  } catch(e) { next(e); }
});
// Fixed upstream host and an explicit route allowlist. Same-origin access works through the Express proxy in development and production.
router.use(async (req,res,next) => {
  const workspace = /^\/workspace\/(catalog|tiles(?:\/[^/]+\/[^/]+(?:\/image|\/evidence)?)?|jobs(?:\/[^/]+\/artifact)?|training|inference)$/;
  const wardPath = /^\/wards(?:\/[1-9]\d*\/(geometry|topology|edit|status|conflicts))?$/;
  if (!workspace.test(req.path) && !wardPath.test(req.path)) { next(); return; }
  const canPost = /^\/workspace\/(inference|training)$/.test(req.path) || /^\/wards\/[1-9]\d*\/edit$/.test(req.path);
  if (req.method !== 'GET' && !(req.method === 'POST' && canPost)) { res.status(405).json({error:'Method not allowed'}); return; }
  try {
    const response = await fetch(`${base}${req.url}`, { ...(req.method === 'POST' ? jsonPost(req.body) : {}), signal:AbortSignal.timeout(30000) });
    res.status(response.status);
    for(const header of ['content-type','content-disposition']) { const value=response.headers.get(header); if(value)res.setHeader(header,value); }
    res.send(Buffer.from(await response.arrayBuffer()));
  } catch(e) { next(e); }
});
router.use((err: unknown,_req: import('express').Request,res:import('express').Response,_next:import('express').NextFunction) => {
  res.status(err instanceof ApiError ? err.status : err instanceof z.ZodError ? 400 : 502).json({error:err instanceof Error?err.message:'Backend unavailable'});
});
export default router;
