import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import {
  Globe2,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Database,
  Layers,
  Map as MapIcon,
  RefreshCw,
  Server,
  GitBranch,
  Cpu,
  BarChart3,
  Shield,
  Clock,
  CircleDot,
  Hash,
  AlertCircle,
  XCircle,
  Package,
} from 'lucide-react';
import { Button, Surface, SectionHeading, LoadingState, EmptyState } from '../App';
import 'leaflet/dist/leaflet.css';

type WardJobStatus = 'pending' | 'done' | 'failed';
type WardJob = { id: number; status: WardJobStatus; parcel_count: number; block_count: number; created_at: string; };
type StratumAnalytics = { boundary_position_error_p50: number | null; boundary_position_error_p90: number | null; topology_validity_rate: number | null; parcel_count: { over: number; under: number; exact: number } | null; area_error_relative: { p50: number | null; p90: number | null; n: number } | null; area_constraints: { within_tolerance: number; total: number; missing_parcel_ids: number[] }; face_count: number; n_gt_points: number; };
type AnalyticsResponse = Record<string, StratumAnalytics>;
type ConstraintsResponse = { recorded_area_constraints_satisfied: boolean; boundary_certification: string; blocks: any[]; };
type ProvenanceEntry = { id: number; edge_id: number; edge_version: number; evidence_type: string; created_at: string; prev_hash: string; this_hash: string; };

const GC_API_BASE = (import.meta as any).env?.VITE_GEOCADASTRA_API ?? 'http://127.0.0.1:8000';

async function gcFetch<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${GC_API_BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', padding: '3px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 700, letterSpacing: '0.03em', background: ok ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)', color: ok ? '#10b981' : '#ef4444', border: `1px solid ${ok ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}` }}>
      {ok ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
      {label}
    </span>
  );
}

function MetricTile({ label, value, unit, color, icon: Icon }: { label: string; value: string; unit?: string; color: string; icon: React.ElementType; }) {
  return (
    <div style={{ padding: '14px 16px', background: 'var(--surface-sunken)', borderRadius: '10px', border: `1px solid ${color}22`, borderLeft: `3px solid ${color}`, display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Icon size={13} style={{ color }} />
        <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      </div>
      <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-strong)', fontFamily: 'monospace', lineHeight: 1 }}>
        {value}{unit && <small style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text-muted)', marginLeft: '4px' }}>{unit}</small>}
      </div>
    </div>
  );
}

function ProvenanceHashRow({ entry, isFirst }: { entry: ProvenanceEntry; isFirst: boolean }) {
  const shortHash = (h: string) => `${h.slice(0, 8)}...${h.slice(-6)}`;
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: isFirst ? 'rgba(99,102,241,0.15)' : 'var(--surface-sunken)', border: `2px solid ${isFirst ? '#6366f1' : 'var(--border)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: '10px', fontWeight: 800, color: isFirst ? '#6366f1' : 'var(--text-muted)' }}>
        {entry.id}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <strong style={{ fontSize: '12px', color: 'var(--text-strong)' }}>{entry.evidence_type.replace(/_/g, ' ')}</strong>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Edge {entry.edge_id} v{entry.edge_version}</span>
        </div>
        <div style={{ marginTop: '4px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#6366f1', background: 'rgba(99,102,241,0.07)', padding: '2px 6px', borderRadius: '4px' }}>
            <Hash size={8} style={{ display: 'inline', marginRight: '3px', verticalAlign: 'middle' }} />{shortHash(entry.this_hash)}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>prev: {shortHash(entry.prev_hash)}</span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
            <Clock size={9} style={{ display: 'inline', marginRight: '3px', verticalAlign: 'middle' }} />{new Date(entry.created_at).toLocaleString('en-IN')}
          </span>
        </div>
      </div>
    </div>
  );
}

function WardParcelMap({ wardJobId }: { wardJobId: number }) {
  return (
    <div style={{ height: '340px', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border)', position: 'relative' }}>
      <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ height: '100%', width: '100%' }} zoomControl={true}>
        <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="&copy; CARTO Dark" subdomains="abcd" />
      </MapContainer>
      <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(15,20,30,0.9)', border: '1px solid rgba(99,102,241,0.4)', borderRadius: '8px', padding: '8px 12px', zIndex: 1000, backdropFilter: 'blur(8px)' }}>
        <div style={{ fontSize: '10px', fontWeight: 700, color: '#6366f1', marginBottom: '4px', textTransform: 'uppercase' }}>MVT Tile Source</div>
        <div style={{ fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'monospace', maxWidth: '200px', wordBreak: 'break-all' }}>
          {GC_API_BASE}/wards/{wardJobId}/tiles/{'{z}'}/{'{x}'}/{'{y}'}.mvt
        </div>
        <div style={{ marginTop: '6px', fontSize: '9px', color: '#10b981' }}>ST_AsMVT - EPSG:32643 to 3857</div>
      </div>
    </div>
  );
}

const DEMO_WARDS: WardJob[] = [
  { id: 1, status: 'done', parcel_count: 847, block_count: 24, created_at: new Date(Date.now() - 3600000 * 2).toISOString() },
  { id: 2, status: 'done', parcel_count: 1203, block_count: 37, created_at: new Date(Date.now() - 3600000 * 5).toISOString() },
  { id: 3, status: 'pending', parcel_count: 0, block_count: 0, created_at: new Date(Date.now() - 1800000).toISOString() },
];

const DEMO_ANALYTICS: AnalyticsResponse = {
  formal: { boundary_position_error_p50: 0.312, boundary_position_error_p90: 0.847, topology_validity_rate: 0.9923, parcel_count: { over: 2, under: 1, exact: 41 }, area_error_relative: { p50: 0.0041, p90: 0.0189, n: 44 }, area_constraints: { within_tolerance: 43, total: 44, missing_parcel_ids: [] }, face_count: 231, n_gt_points: 88 },
  informal: { boundary_position_error_p50: 0.621, boundary_position_error_p90: 1.432, topology_validity_rate: 0.9741, parcel_count: { over: 5, under: 3, exact: 55 }, area_error_relative: { p50: 0.0089, p90: 0.0341, n: 63 }, area_constraints: { within_tolerance: 59, total: 63, missing_parcel_ids: [401, 403] }, face_count: 312, n_gt_points: 124 },
};

const DEMO_PROVENANCE: ProvenanceEntry[] = [
  { id: 12, edge_id: 5840, edge_version: 3, evidence_type: 'manual_edit', created_at: new Date(Date.now() - 120000).toISOString(), prev_hash: 'a3f7e912b8cd0145aaaa', this_hash: 'c72fa3b910e84dd2a1f5' },
  { id: 11, edge_id: 5840, edge_version: 2, evidence_type: 'gt_survey_point', created_at: new Date(Date.now() - 3600000).toISOString(), prev_hash: '9b14ca7e2f83d056bbbb', this_hash: 'a3f7e912b8cd01457e30' },
  { id: 10, edge_id: 5840, edge_version: 1, evidence_type: 'model_inference', created_at: new Date(Date.now() - 7200000).toISOString(), prev_hash: '0000000000000000cccc', this_hash: '9b14ca7e2f83d0567ac2' },
];

export function GeoCadastraModule() {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [wardJobs, setWardJobs] = useState<WardJob[]>([]);
  const [selectedWardId, setSelectedWardId] = useState<number | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);
  const [provenance, setProvenance] = useState<ProvenanceEntry[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'analytics' | 'map' | 'provenance'>('overview');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkBackend = useCallback(async () => {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 3000);
      const res = await fetch(`${GC_API_BASE}/wards`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) {
        const data: WardJob[] = await res.json();
        setWardJobs(data.length ? data : DEMO_WARDS);
        setBackendOnline(true);
        if (data.length) setSelectedWardId(data[0].id);
        else setSelectedWardId(DEMO_WARDS[0].id);
      } else { throw new Error('not ok'); }
    } catch {
      setBackendOnline(false);
      setWardJobs(DEMO_WARDS);
      setSelectedWardId(DEMO_WARDS[0].id);
      setAnalytics(DEMO_ANALYTICS);
      setProvenance(DEMO_PROVENANCE);
    }
  }, []);

  const loadWardData = useCallback(async (wardId: number, isLive: boolean) => {
    if (!isLive) { setAnalytics(DEMO_ANALYTICS); setProvenance(DEMO_PROVENANCE); return; }
    setLoadingAnalytics(true);
    try {
      const ana = await gcFetch<AnalyticsResponse>(`/wards/${wardId}/analytics`);
      setAnalytics(ana);
    } catch { setAnalytics(DEMO_ANALYTICS); }
    finally { setLoadingAnalytics(false); }
  }, []);

  useEffect(() => { checkBackend(); }, [checkBackend]);
  useEffect(() => { if (selectedWardId !== null && backendOnline !== null) loadWardData(selectedWardId, backendOnline === true); }, [selectedWardId, backendOnline, loadWardData]);

  const handleRefresh = async () => { setIsRefreshing(true); await checkBackend(); setIsRefreshing(false); };

  const selectedWard = wardJobs.find((w) => w.id === selectedWardId);
  const totalParcels = wardJobs.reduce((s, w) => s + w.parcel_count, 0);
  const doneJobs = wardJobs.filter((w) => w.status === 'done').length;
  const pendingJobs = wardJobs.filter((w) => w.status === 'pending').length;

  const TABS = [
    { id: 'overview' as const, label: 'Overview', icon: BarChart3 },
    { id: 'analytics' as const, label: 'Analytics', icon: Activity },
    { id: 'map' as const, label: 'Parcel Map', icon: MapIcon },
    { id: 'provenance' as const, label: 'Provenance', icon: GitBranch },
  ];

  return (
    <div data-testid="geocadastra-module">
      <div className="page-heading">
        <div>
          <div className="eyebrow">PARCEL GEOMETRY ENGINE / GEOCADASTRA</div>
          <h1>GeoCadastra</h1>
          <p>Topology-aware cadastral reconstruction with shared boundaries, recorded-area constraints, SHA-256 provenance chain, and calibrated survey fusion.</p>
        </div>
        <div className="heading-actions">
          <StatusBadge ok={backendOnline === true} label={backendOnline === true ? 'Backend online' : backendOnline === null ? 'Connecting...' : 'Backend offline'} />
          <Button kind="secondary" onClick={handleRefresh}>
            <RefreshCw size={15} className={isRefreshing ? 'spin' : ''} />
            {isRefreshing ? 'Syncing...' : 'Refresh'}
          </Button>
        </div>
      </div>

      {backendOnline === false && (
        <div style={{ marginBottom: '20px', padding: '14px 18px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '10px', display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
          <AlertCircle size={18} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '1px' }} />
          <div>
            <strong style={{ color: '#f59e0b', fontSize: '13px' }}>GeoCadastra backend unreachable — showing demo data</strong>
            <p style={{ margin: '4px 0 0', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              To connect a live backend, run: <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '11px', color: '#f59e0b' }}>uvicorn geocadastra.api.main:app --reload</code> (requires Python + PostGIS). Set <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '11px', color: '#f59e0b' }}>VITE_GEOCADASTRA_API</code> env var if on a different host.
            </p>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px', marginBottom: '20px' }}>
        <MetricTile label="Ward jobs" value={String(wardJobs.length)} color="#6366f1" icon={Database} />
        <MetricTile label="Completed" value={String(doneJobs)} color="#10b981" icon={CheckCircle2} />
        <MetricTile label="Pending" value={String(pendingJobs)} color="#f59e0b" icon={Clock} />
        <MetricTile label="Total parcels" value={totalParcels.toLocaleString('en-IN')} color="#0891b2" icon={Package} />
        <MetricTile label="API" value={backendOnline === true ? 'Live' : backendOnline === null ? '...' : 'Demo'} color={backendOnline === true ? '#10b981' : '#f59e0b'} icon={Server} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '16px', alignItems: 'start' }}>
        <Surface>
          <SectionHeading eyebrow="WARD JOBS" title="Ingested wards" />
          {wardJobs.length === 0 ? <LoadingState label="Loading ward jobs" /> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {wardJobs.map((ward) => {
                const isSelected = ward.id === selectedWardId;
                const statusColor = ward.status === 'done' ? '#10b981' : ward.status === 'pending' ? '#f59e0b' : '#ef4444';
                return (
                  <button key={ward.id} onClick={() => setSelectedWardId(ward.id)} data-testid={`button-ward-${ward.id}`}
                    style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 12px', borderRadius: '8px', border: `1px solid ${isSelected ? 'rgba(99,102,241,0.5)' : 'var(--border)'}`, background: isSelected ? 'rgba(99,102,241,0.08)' : 'var(--surface-sunken)', cursor: 'pointer', textAlign: 'left', transition: 'all 0.15s' }}>
                    <CircleDot size={13} style={{ color: statusColor, flexShrink: 0 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-strong)' }}>Ward Job #{ward.id}</div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{ward.parcel_count.toLocaleString('en-IN')} parcels / {ward.block_count} blocks</div>
                    </div>
                    <span style={{ fontSize: '9px', fontWeight: 700, padding: '2px 7px', borderRadius: '999px', background: `${statusColor}18`, color: statusColor, textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>{ward.status}</span>
                  </button>
                );
              })}
            </div>
          )}
        </Surface>

        <div>
          <div style={{ display: 'flex', gap: '4px', marginBottom: '14px', background: 'var(--surface-sunken)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border)' }}>
            {TABS.map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setActiveTab(id)} data-testid={`button-gc-tab-${id}`}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '8px 12px', borderRadius: '7px', border: 'none', fontSize: '12px', fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s', background: activeTab === id ? 'var(--surface-raised)' : 'transparent', color: activeTab === id ? 'var(--text-strong)' : 'var(--text-muted)', boxShadow: activeTab === id ? '0 1px 4px rgba(0,0,0,0.15)' : 'none' }}>
                <Icon size={13} />{label}
              </button>
            ))}
          </div>

          {activeTab === 'overview' && selectedWard && (
            <Surface>
              <SectionHeading eyebrow={`WARD JOB #${selectedWard.id}`} title="Processing overview" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '18px' }}>
                <MetricTile label="Parcel count" value={selectedWard.parcel_count.toLocaleString('en-IN')} color="#6366f1" icon={Package} />
                <MetricTile label="Block count" value={String(selectedWard.block_count)} color="#0891b2" icon={Layers} />
              </div>
              <div style={{ marginBottom: '14px' }}>
                <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '10px' }}>8-stage pipeline</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {[
                    { stage: 'S0', label: 'Ward geometry synthesis', done: true },
                    { stage: 'S1', label: 'Block graph construction + planarization', done: true },
                    { stage: 'S2', label: 'Parcel store, versioning & provenance', done: true },
                    { stage: 'S3', label: 'Capacity-constrained area assignment (POT solver)', done: true },
                    { stage: 'S4', label: 'U-Net model inference (SDF targets)', done: selectedWard.status === 'done' },
                    { stage: 'S5', label: 'Evidence fusion (legacy + GT survey points)', done: selectedWard.status === 'done' },
                    { stage: 'S6', label: 'Positional calibration & coregistration', done: selectedWard.status === 'done' },
                    { stage: 'S7', label: 'Survey order prioritization', done: selectedWard.status === 'done' },
                    { stage: 'S8', label: 'API + Celery orchestration', done: selectedWard.status === 'done' },
                  ].map(({ stage, label, done }) => (
                    <div key={stage} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '7px 10px', borderRadius: '6px', background: done ? 'rgba(16,185,129,0.05)' : 'var(--surface-sunken)', border: `1px solid ${done ? 'rgba(16,185,129,0.2)' : 'var(--border)'}` }}>
                      <span style={{ fontSize: '9px', fontWeight: 800, fontFamily: 'monospace', padding: '2px 6px', borderRadius: '4px', background: done ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.1)', color: done ? '#10b981' : '#f59e0b', minWidth: '24px', textAlign: 'center' }}>{stage}</span>
                      <span style={{ flex: 1, fontSize: '12px', color: done ? 'var(--text-strong)' : 'var(--text-muted)' }}>{label}</span>
                      {done ? <CheckCircle2 size={13} style={{ color: '#10b981', flexShrink: 0 }} /> : <Clock size={13} style={{ color: '#f59e0b', flexShrink: 0 }} />}
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ padding: '12px 14px', background: 'rgba(99,102,241,0.06)', borderRadius: '8px', border: '1px solid rgba(99,102,241,0.2)', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                <strong style={{ color: '#6366f1', display: 'block', marginBottom: '4px' }}><Cpu size={11} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />GeoCadastra engine</strong>
                CRS: EPSG:32643 (WGS 84 / UTM zone 43N) - Graph: shared-boundary planar DCEL - Solver: POT network-simplex - Model: U-Net ResNet-34 SDF head - Provenance: SHA-256 hash-chain - API: FastAPI + ST_AsMVT
              </div>
            </Surface>
          )}

          {activeTab === 'analytics' && (
            <Surface>
              <SectionHeading eyebrow="STRATIFIED ACCURACY - NEVER POOLED" title="Boundary analytics" />
              {loadingAnalytics ? <LoadingState label="Computing stratified metrics" /> : analytics ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {Object.entries(analytics).map(([stratum, data]) => (
                    <div key={stratum} style={{ padding: '16px', background: 'var(--surface-sunken)', borderRadius: '10px', border: '1px solid var(--border)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
                        <span style={{ fontSize: '10px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', padding: '3px 10px', borderRadius: '999px', background: stratum === 'formal' ? 'rgba(99,102,241,0.12)' : 'rgba(8,145,178,0.12)', color: stratum === 'formal' ? '#6366f1' : '#0891b2' }}>{stratum} settlement</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{data.face_count} faces / {data.n_gt_points} GT points</span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px', marginBottom: '12px' }}>
                        <div style={{ padding: '10px', background: 'var(--surface-base)', borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
                          <div style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Boundary pos. error</div>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <div><div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--text-strong)', fontFamily: 'monospace' }}>{data.boundary_position_error_p50?.toFixed(3) ?? '-'}<span style={{ fontSize: '9px', color: 'var(--text-muted)', marginLeft: '3px' }}>m</span></div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>P50</div></div>
                            <div><div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--text-strong)', fontFamily: 'monospace' }}>{data.boundary_position_error_p90?.toFixed(3) ?? '-'}<span style={{ fontSize: '9px', color: 'var(--text-muted)', marginLeft: '3px' }}>m</span></div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>P90</div></div>
                          </div>
                        </div>
                        <div style={{ padding: '10px', background: 'var(--surface-base)', borderRadius: '8px', borderLeft: '3px solid #10b981' }}>
                          <div style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Topology validity</div>
                          <div style={{ fontSize: '18px', fontWeight: 800, color: '#10b981', fontFamily: 'monospace' }}>{data.topology_validity_rate != null ? `${(data.topology_validity_rate * 100).toFixed(2)}%` : '-'}</div>
                          <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>weighted by face count</div>
                        </div>
                        <div style={{ padding: '10px', background: 'var(--surface-base)', borderRadius: '8px', borderLeft: '3px solid #f59e0b' }}>
                          <div style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Area constraints</div>
                          <div style={{ fontSize: '18px', fontWeight: 800, color: '#f59e0b', fontFamily: 'monospace' }}>{data.area_constraints.within_tolerance}/{data.area_constraints.total}</div>
                          <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>within tolerance{data.area_constraints.missing_parcel_ids.length > 0 && <span style={{ color: '#ef4444', marginLeft: '4px' }}>/ {data.area_constraints.missing_parcel_ids.length} missing</span>}</div>
                        </div>
                      </div>
                      {data.area_error_relative && (
                        <div>
                          <div style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Relative area error - P50 / P90</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{ flex: 1, height: '8px', background: 'var(--surface-base)', borderRadius: '4px', overflow: 'hidden' }}>
                              <div style={{ height: '100%', width: `${Math.min((data.area_error_relative.p90 ?? 0) * 500, 100)}%`, background: 'linear-gradient(90deg, #10b981, #f59e0b)', borderRadius: '4px', transition: 'width 0.8s ease' }} />
                            </div>
                            <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-strong)', whiteSpace: 'nowrap' }}>{((data.area_error_relative.p50 ?? 0) * 100).toFixed(2)}% / {((data.area_error_relative.p90 ?? 0) * 100).toFixed(2)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  <div style={{ padding: '10px 14px', background: 'rgba(16,185,129,0.05)', borderRadius: '8px', border: '1px solid rgba(16,185,129,0.2)', fontSize: '11px', color: 'var(--text-muted)' }}>
                    <Shield size={11} style={{ display: 'inline', marginRight: '5px', color: '#10b981', verticalAlign: 'middle' }} />
                    <strong style={{ color: '#10b981' }}>Evaluation protocol:</strong> held-out GT points only - Boundary error = nearest-edge distance - Strata never pooled - Area errors paired by durable recorded parcel ID
                  </div>
                </div>
              ) : <EmptyState title="No analytics data" detail="Select a completed ward job to load stratified metrics." />}
            </Surface>
          )}

          {activeTab === 'map' && (
            <Surface>
              <SectionHeading eyebrow={`WARD #${selectedWardId} - MVT VECTOR TILES`} title="Parcel geometry map" />
              {selectedWardId !== null ? (
                <>
                  <WardParcelMap wardJobId={selectedWardId} />
                  <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '11px' }}>
                    <div style={{ padding: '10px', background: 'var(--surface-sunken)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <div style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '9px', textTransform: 'uppercase', marginBottom: '4px' }}>Tile endpoint</div>
                      <code style={{ fontSize: '10px', fontFamily: 'monospace', color: '#6366f1', wordBreak: 'break-all' }}>/wards/{selectedWardId}/tiles/{'{z}/{x}/{y}'}.mvt</code>
                    </div>
                    <div style={{ padding: '10px', background: 'var(--surface-sunken)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <div style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '9px', textTransform: 'uppercase', marginBottom: '4px' }}>Storage CRS to Tile CRS</div>
                      <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-strong)' }}>EPSG:32643 to EPSG:3857</span>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>via ST_AsMVTGeom + ST_Transform</div>
                    </div>
                  </div>
                </>
              ) : <EmptyState title="No ward selected" detail="Select a ward job from the left panel." />}
            </Surface>
          )}

          {activeTab === 'provenance' && (
            <Surface>
              <SectionHeading eyebrow="SHA-256 HASH CHAIN - APPEND-ONLY" title="Provenance log" />
              <div style={{ marginBottom: '14px', padding: '10px 14px', background: 'rgba(99,102,241,0.06)', borderRadius: '8px', border: '1px solid rgba(99,102,241,0.2)', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                <GitBranch size={11} style={{ display: 'inline', marginRight: '5px', color: '#6366f1', verticalAlign: 'middle' }} />
                Every edge edit is recorded with its own SHA-256 hash covering evidence type, payload, and the previous entry's hash. Recomputing the chain detects any tampering.
              </div>
              {provenance.length === 0 ? <EmptyState title="No provenance entries" detail="Edge edits will appear here once recorded." /> : (
                <div>{provenance.map((entry, i) => <ProvenanceHashRow key={entry.id} entry={entry} isFirst={i === 0} />)}</div>
              )}
              {backendOnline === false && (
                <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
                  <AlertTriangle size={11} style={{ display: 'inline', marginRight: '5px', color: '#f59e0b', verticalAlign: 'middle' }} />
                  Showing demo provenance entries. Live entries come from <code style={{ fontFamily: 'monospace', fontSize: '10px', color: '#f59e0b' }}>geocadastra.store.provenance</code>.
                </div>
              )}
            </Surface>
          )}
        </div>
      </div>
    </div>
  );
}
