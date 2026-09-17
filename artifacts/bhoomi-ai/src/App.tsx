import GeoVlmSentinelModule from "./components/GeoVlmSentinelModule";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState, useEffect, useRef } from 'react';
import { Link, Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import { IndiaBoundaryMap } from './components/IndiaBoundaryMap';
import { ParcelExplorerModal } from './components/ParcelExplorerModal';
import { TopologyModule as Topology } from './components/TopologyModule';
import { Dashboard, Ingestion } from './components/RealWorkspace';
import { ParcelMap } from './components/ParcelExplorerModal';
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpRight,
  BadgeCheck,
  BatteryMedium,
  Bell,
  Brain,
  Building2,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  CloudOff,
  Download,
  FileArchive,
  FileCheck2,
  FileUp,
  FlaskConical,
  Hammer,
  Layers3,
  Map,
  MapPinned,
  Menu,
  MoreHorizontal,
  Network,
  PackageCheck,
  PanelLeft,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Satellite,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  UploadCloud,
  UserRound,
  WifiOff,
  X,
  Zap,
} from 'lucide-react';
import {
  getGetParcelQueryKey,
  useCreateExport,
  useCreateProcessingRun,
  useFixTopology,
  useGetDashboard,
  useGetParcel,
  useListChanges,
  useListParcels,
  useListProcessingRuns,
  useListRegions,
  useScanTopology,
  useUpdateParcel,
} from '@workspace/api-client-react';
import type {
  ChangeDetection,
  DashboardSummary,
  Parcel,
  ProcessingRun,
  Region,
  TopologyReport,
} from '@workspace/api-client-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import NotFound from '@/pages/not-found';
import './index.css';

const queryClient = new QueryClient();

const nav = [
  { href: '/', label: 'Command center', icon: Radio },
  { href: '/ingestion', label: 'Inference studio', icon: UploadCloud },
  { href: '/topology', label: 'Topology', icon: Network },
  { href: '/sentinel', label: 'Geo-VLM Sentinel', icon: Brain },
  { href: '/changes', label: 'Change detection', icon: Layers3 },
  { href: '/field', label: 'Field verification', icon: MapPinned },
  { href: '/exports', label: 'Export center', icon: ArrowDownToLine },
  { href: '/billing', label: 'Billing', icon: FileCheck2 },
];

function formatNumber(value: number | null | undefined, digits = 0) {
  return value == null ? 'Not measured' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(value);
}

function timeAgo(value: string | undefined) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  const minutes = Math.max(1, Math.round((Date.now() - date.valueOf()) / 60000));
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  return `${Math.round(hours / 24)} d ago`;
}

function toneClass(tone: string | undefined) {
  if (tone === 'warning' || tone === 'high') return 'tone-warn';
  if (tone === 'success' || tone === 'resolved') return 'tone-good';
  if (tone === 'critical') return 'tone-danger';
  return 'tone-info';
}

function statusClass(status: string | undefined) {
  const value = (status ?? '').toLowerCase();
  if (value.includes('complete') || value.includes('valid') || value.includes('verified') || value.includes('ready')) return 'status-good';
  if (value.includes('error') || value.includes('blocked') || value.includes('critical')) return 'status-danger';
  if (value.includes('review') || value.includes('pending') || value.includes('running')) return 'status-warn';
  return 'status-neutral';
}

export function Surface({ children, className = '', style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return <section style={style} className={`surface ${className}`}>{children}</section>;
}

export function SectionHeading({ eyebrow, title, action }: { eyebrow?: string; title: string; action?: React.ReactNode }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

export function Button({
  children,
  onClick,
  kind = 'primary',
  type = 'button',
  disabled = false,
  className = '',
  style,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  kind?: 'primary' | 'secondary' | 'ghost' | 'danger';
  type?: 'button' | 'submit';
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <button style={style} data-testid={`button-${String(children).replace(/\s+/g, '-').toLowerCase()}`} className={`button button-${kind} ${className}`} onClick={onClick} type={type} disabled={disabled}>
      {children}
    </button>
  );
}

export function LoadingState({ label = 'Synchronizing command data' }: { label?: string }) {
  return (
    <div className="loading-stack" data-testid="status-loading">
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-panel" />
      <p>{label} <span className="pulse-dot" /></p>
    </div>
  );
}

function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="empty-state error-state" data-testid="status-error">
      <AlertTriangle size={20} />
      <div><strong>Signal interrupted</strong><p>We could not retrieve this operational layer.</p></div>
      {onRetry && <Button kind="secondary" onClick={onRetry}>Retry</Button>}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state" data-testid="status-empty">
      <CircleDot size={20} />
      <div><strong>{title}</strong><p>{detail}</p></div>
    </div>
  );
}

function Header({ onMobileMenu }: { onMobileMenu: () => void }) {
  const [location] = useLocation();
  const current = nav.find((item) => item.href === location)?.label ?? 'Command center';
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }).toUpperCase());
    };
    updateTime();
    const timer = setInterval(updateTime, 10000); // update every 10 seconds
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onMobileMenu} data-testid="button-open-navigation"><Menu size={20} /></button>
      <div className="crumb"><span className="crumb-root">BhoomiDrishti AI</span><ChevronRight size={13} /><span>{current}</span></div>
      <div className="topbar-tagline">भूमेः दृष्टिः, सीमायाः सटीकता।</div>
      <div className="topbar-tools">
        <div className="sync-status"><span className="live-dot" /> Live sync <span className="sync-time">{timeStr}</span></div>
      </div>
    </header>
  );
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [location] = useLocation();
  return (
    <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
      <div className="brand">
        <div className="brand-mark"><Target size={21} strokeWidth={2.4} /></div>
        <div><strong>BhoomiDrishti<span>AI</span></strong><small>भूमेः दृष्टिः · Land intelligence</small></div>
        <button className="sidebar-close" onClick={onClose} data-testid="button-close-navigation"><X size={18} /></button>
      </div>
      <nav className="nav">
        <div className="nav-label">Operational layers</div>
        {nav.slice(0, 6).map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} onClick={onClose} className={`nav-item ${location === href ? 'nav-active' : ''}`} data-testid={`link-${label.toLowerCase().replace(/\s+/g, '-')}`}>
            <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{href === '/' && <span className="nav-pulse" />}
          </Link>
        ))}
        <div className="nav-label nav-label-secondary">Governance</div>
        {nav.slice(6).map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} onClick={onClose} className={`nav-item ${location === href ? 'nav-active' : ''}`} data-testid={`link-${label.toLowerCase().replace(/\s+/g, '-')}`}>
            <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="sidebar-foot">
        <div className="foot-status"><span className="live-dot" /><span>All systems nominal</span><span className="mono">99.98%</span></div>
        <div className="sidebar-meta">DoLR / Municipal GIS<br />Build 2.6.14 · Secure workspace</div>
      </div>
    </aside>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="app-shell">
      <Sidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />
      {mobileOpen && <button className="sidebar-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close navigation" data-testid="button-navigation-backdrop" />}
      <main className="main"><Header onMobileMenu={() => setMobileOpen(true)} /><div className="page">{children}</div></main>
    </div>
  );
}

function KPICard({ label, value, suffix, trend, accent, detail }: { label: string; value: string; suffix?: string; trend?: string; accent: string; detail: string }) {
  return (
    <div className="kpi-card" data-testid={`metric-${label.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="kpi-top"><span>{label}</span><span className={`kpi-accent ${accent}`} /></div>
      <div className="kpi-value">{value}{suffix && <small>{suffix}</small>}</div>
      <div className="kpi-foot"><span className="kpi-detail">{detail}</span>{trend && <span className="kpi-trend"><ArrowUpRight size={13} /> {trend}</span>}</div>
    </div>
  );
}

function MiniMap({ parcels = [] }: { parcels?: Parcel[]; focusedId?: string }) {
  return <ParcelMap parcels={parcels}/>;
}

function Changes() {
  const changes = useListChanges();
  const [filter, setFilter] = useState('all');
  const data = (changes.data as ChangeDetection[] | undefined) ?? [];
  const filtered = filter === 'all' ? data : data.filter((item) => item.severity.toLowerCase() === filter);
  return (
    <>
      <div className="page-heading"><div><div className="eyebrow">TEMPORAL INTELLIGENCE / 2024—25</div><h1>Change detection</h1><p>Review new construction, land-use drift and possible encroachment.</p></div><div className="heading-actions"><Button kind="secondary" onClick={() => changes.refetch()}><RefreshCw size={15} /> Refresh detections</Button></div></div>
      <div className="change-summary"><div><span className="summary-overline">UNRESOLVED SIGNALS</span><strong>{formatNumber(data.length)}</strong><small>detected in last 30 days</small></div><div><span className="summary-overline">HIGH SEVERITY</span><strong className="text-danger">{formatNumber(data.filter((item) => item.severity.toLowerCase() === 'high').length)}</strong><small>needs officer review</small></div><div><span className="summary-overline">AREA FLAGGED</span><strong>{formatNumber(data.reduce((sum, item) => sum + item.areaSqM, 0))}<small> m²</small></strong><small>across {new Set(data.map((item) => item.regionName)).size} regions</small></div><div className="change-graphic"><div className="change-ring"><span>30d</span></div><span>temporal window</span></div></div>
      <Surface><SectionHeading eyebrow="REVIEW QUEUE" title="Detected changes" action={<div className="filter-tabs">{['all', 'high', 'medium', 'low'].map((item) => <button key={item} className={filter === item ? 'filter-active' : ''} onClick={() => setFilter(item)} data-testid={`button-filter-${item}`}>{item}</button>)}</div>} />{changes.isLoading ? <LoadingState label="Comparing temporal layers" /> : changes.isError ? <ErrorState onRetry={() => changes.refetch()} /> : filtered.length ? <div className="change-list">{filtered.map((item) => <ChangeRow key={item.id} item={item} />)}</div> : <EmptyState title="No changes in this filter" detail="The current temporal layer has no detections matching this view." />}</Surface>
    </>
  );
}

function ChangeRow({ item }: { item: ChangeDetection }) {
  return <div className="change-row" data-testid={`row-change-${item.id}`}><div className={`change-severity severity-${item.severity.toLowerCase()}`}><AlertTriangle size={16} /></div><div className="change-title"><strong>{item.title}</strong><span>{item.category} · {item.regionName}</span></div><div className="change-area"><small>AFFECTED AREA</small><strong>{formatNumber(item.areaSqM)} m²</strong></div><div className="change-date"><small>DETECTED</small><span>{timeAgo(item.detectedAt)}</span></div><span className={`status-pill ${item.severity.toLowerCase() === 'high' ? 'status-danger' : 'status-warn'}`}>{item.severity}</span><button className="icon-button subtle" data-testid={`button-change-actions-${item.id}`}><MoreHorizontal size={17} /></button></div>;
}

function Field() {
  const parcels = useListParcels({ status: 'pending' });
  const update = useUpdateParcel();
  const [selectedId, setSelectedId] = useState('');
  const parcelList = (parcels.data as Parcel[] | undefined) ?? [];
  const selectedFromList = parcelList.find((item) => item.id === selectedId);
  const detail = useGetParcel(selectedId || '', { query: { enabled: !!selectedId, queryKey: getGetParcelQueryKey(selectedId || '') } });
  const selected = (detail.data as Parcel | undefined) ?? selectedFromList;
  const verify = () => { if (selected) update.mutate({ id: selected.id, data: { status: 'verified' } }); };
  return (
    <>
      <div className="page-heading"><div><div className="eyebrow">FIELD OPERATIONS / OFFLINE-FIRST</div><h1>Ground-truthing desk</h1><p>Resolve uncertain parcels with evidence from the field, even without signal.</p></div><div className="heading-actions"><span className="offline-chip"><WifiOff size={14} /> Offline queue <strong>12</strong></span><Button kind="secondary" onClick={() => parcels.refetch()}><RefreshCw size={15} /> Sync when online</Button></div></div>
      <div className="field-banner"><div className="field-signal"><BatteryMedium size={20} /><span><strong>Field network unavailable</strong><small>Last sync 18 minutes ago · work continues locally</small></span></div><div className="field-progress"><span>12 queued records</span><div><i style={{ width: '64%' }} /></div><small>64% of today’s queue reviewed</small></div></div>
      <div className="field-layout"><Surface className="field-list"><SectionHeading eyebrow="VERIFICATION QUEUE" title="Parcels awaiting evidence" action={<span className="mono">{parcelList.length} RECORDS</span>} />{parcels.isLoading ? <LoadingState label="Loading local queue" /> : parcels.isError ? <ErrorState onRetry={() => parcels.refetch()} /> : parcelList.length ? <div className="parcel-list">{parcelList.map((parcel) => <button key={parcel.id} className={`parcel-row ${selectedId === parcel.id ? 'parcel-row-active' : ''}`} onClick={() => setSelectedId(parcel.id)} data-testid={`button-select-parcel-${parcel.id}`}><span className="parcel-status"><CircleDot size={14} /></span><span><strong>{parcel.ulpin}</strong><small>{parcel.regionName} · {formatNumber(parcel.areaSqM)} m²</small></span><span className="confidence">{formatNumber(parcel.confidence, 0)}%<small>confidence</small></span><ChevronRight size={16} /></button>)}</div> : <EmptyState title="Queue is clear" detail="No pending parcel records need ground-truthing." />}</Surface><Surface className="field-detail">{selected ? <><div className="detail-header"><div><div className="eyebrow">PARCEL RECORD</div><h2>{selected.ulpin}</h2><p>{selected.regionName} · Updated {timeAgo(selected.updatedAt)}</p></div><span className={`status-pill ${statusClass(selected.status)}`}>{selected.status}</span></div><MiniMap parcels={[selected]} focusedId={selected.id} /><div className="detail-facts"><div><small>AREA</small><strong>{formatNumber(selected.areaSqM)} m²</strong></div><div><small>MODEL CONFIDENCE</small><strong>{formatNumber(selected.confidence, 1)}%</strong></div><div><small>OWNERSHIP</small><strong>{selected.ownership}</strong></div></div><div className="field-actions"><Button onClick={verify} disabled={update.isPending}><CheckCircle2 size={15} /> {update.isPending ? 'Saving locally' : 'Mark verified'}</Button><Button kind="secondary"><CloudOff size={15} /> Add field note</Button></div></> : <div className="detail-empty"><MapPinned size={30} /><strong>Select a parcel to inspect</strong><p>Choose a record from the offline queue to load geometry and evidence controls.</p></div>}</Surface></div>
    </>
  );
}

function Exports() {
  const regions = useListRegions();
  const create = useCreateExport();
  const [format, setFormat] = useState('GeoPackage');
  const [regionId, setRegionId] = useState('');
  const [created, setCreated] = useState<string[]>([]);
  const regionList = Array.isArray(regions.data) ? (regions.data as Region[]) : [];
  const submit = () => { const id = regionId || regionList[0]?.id; if (!id) return; create.mutate({ data: { format, regionId: id } }, { onSuccess: (job: { fileName: string }) => setCreated((old) => [job.fileName, ...old]) }); };
  return (
    <>
      <div className="page-heading"><div><div className="eyebrow">DELIVERY / REGISTER-READY OUTPUT</div><h1>Export center</h1><p>Package trusted geometry and its chain of custody for downstream systems.</p></div><div className="heading-actions"><div className="security-mark"><ShieldCheck size={15} /> ULPIN compliant</div></div></div>
      {created.length > 0 && <div className="toast-inline" data-testid="status-export-created"><PackageCheck size={16} /> Package created: {created[0]}</div>}
      <div className="export-grid"><Surface className="export-builder"><SectionHeading eyebrow="NEW PACKAGE" title="Build a GIS delivery" /><div className="export-option-group"><span className="field-label">OUTPUT FORMAT</span><div className="format-grid">{['GeoPackage', 'GeoJSON', 'Shapefile', 'ULPIN package'].map((item) => <button key={item} className={`format-option ${format === item ? 'format-selected' : ''}`} onClick={() => setFormat(item)} data-testid={`button-format-${item.toLowerCase().replace(/\s+/g, '-')}`}><FileArchive size={18} /><strong>{item}</strong><small>{item === 'ULPIN package' ? 'Register-ready' : 'GIS-ready layer'}</small>{format === item && <CheckCircle2 size={15} />}</button>)}</div></div><label className="field-label">SOURCE JURISDICTION<select value={regionId} onChange={(event) => setRegionId(event.target.value)} data-testid="select-export-region"><option value="">Select region</option>{regionList.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><div className="package-checklist"><div><Check size={14} /> Geometry validated</div><div><Check size={14} /> ULPIN identifiers attached</div><div><Check size={14} /> Provenance manifest included</div></div><Button onClick={submit} disabled={create.isPending || !regionList.length}>{create.isPending ? <><RefreshCw size={15} className="spin" /> Assembling</> : <><Download size={15} /> Create export package</>}</Button></Surface><Surface className="export-info"><div className="orbit-art"><div className="orbit orbit-a" /><div className="orbit orbit-b" /><div className="orbit-core"><Target size={24} /></div></div><div className="eyebrow">CHAIN OF CUSTODY</div><h2>Nothing leaves unmarked.</h2><p>Every delivery carries the source run, validation state and ULPIN coverage needed for an accountable hand-off.</p><div className="info-line"><span>Manifest signing</span><strong>Ed25519</strong></div><div className="info-line"><span>Coordinate system</span><strong>EPSG:4326</strong></div></Surface></div>
      <Surface><SectionHeading eyebrow="DELIVERY HISTORY" title="Generated packages" /><div className="empty-history">{created.length ? created.map((name, index) => <div className="export-history-row" key={`${name}-${index}`}><FileArchive size={18} /><div><strong>{name}</strong><small>created just now · signed manifest attached</small></div><span className="status-pill status-good">ready</span><Download size={16} /></div>) : <EmptyState title="No packages generated" detail="Create a package when a validated region is ready to move downstream." />}</div></Surface>
    </>
  );
}

function Billing() {
  return (
    <>
      <div className="page-heading"><div><div className="eyebrow">SYSTEM MONITORING / INFRASTRUCTURE</div><h1>Compute & Storage Costs</h1><p>Real-time resource utilization and API consumption for BhoomiDrishti AI.</p></div><div className="heading-actions"><Button kind="secondary"><Send size={15} /> Export Usage Report</Button></div></div>
      <div className="billing-hero"><div><span className="plan-kicker">ACTIVE WORKSPACE · SMART INDIA HACKATHON 2026</span><h2>SIH Developer Tier</h2><p>Provisioned for high-performance GeoAI inference and large-scale spatial data processing.</p><div className="plan-tags"><span><ShieldCheck size={14} /> AWS p4d.24xlarge (GPU)</span><span><Satellite size={14} /> 500GB NVMe SSD</span><span><Map size={14} /> Carto Maps API</span></div></div><div className="price-block"><small>ESTIMATED HOURLY BURN</small><strong>₹312.50</strong><span>based on current usage</span></div></div>
      <div className="billing-grid"><Surface><SectionHeading eyebrow="CURRENT UTILIZATION" title="Live Resource Metrics" /><div className="usage-list"><Usage label="Svamitva Dataset Storage (S3)" value="4.7" suffix=" / 10 GB" percent={47} /><Usage label="GPU Inference Time (U-Net)" value="3.2" suffix=" / 10 Hours" percent={32} /><Usage label="Carto Map Tile Requests" value="1.2k" suffix=" / 10k Limit" percent={12} /></div></Surface><Surface><SectionHeading eyebrow="INFRASTRUCTURE" title="Deployed Services" /><div className="account-list"><div><span className="account-icon"><UserRound size={15} /></span><span><strong>Cloud Provider</strong><small>AWS (ap-south-1 Mumbai Region)</small></span><ChevronRight size={16} /></div><div><span className="account-icon"><FileCheck2 size={15} /></span><span><strong>Model Server</strong><small>PyTorch / FastAPI Endpoint · Active</small></span><ChevronRight size={16} /></div><div><span className="account-icon"><Bell size={15} /></span><span><strong>Frontend Hosting</strong><small>Vite + React (Edge Network)</small></span><ChevronRight size={16} /></div></div></Surface></div>
      <Surface className="invoice-surface"><SectionHeading eyebrow="LOGS" title="Recent Compute Sessions" action={<Button kind="ghost"><Download size={15} /> Download Logs</Button>} /><div className="invoice-row"><span className="invoice-date">13 SEP 2026</span><div><strong>Batch Inference: Rajasthan Jurisdiction</strong><small>Processed 690 patches · 32 mins GPU time</small></div><strong>₹166.00</strong><span className="status-pill status-good">completed</span><Download size={16} /></div><div className="invoice-row"><span className="invoice-date">13 SEP 2026</span><div><strong>Dataset Ingestion & Tiling</strong><small>Svamitva Dataset (4.7 GB) extraction and chunking</small></div><strong>₹45.00</strong><span className="status-pill status-good">completed</span><Download size={16} /></div></Surface>
    </>
  );
}

function Usage({ label, value, suffix, percent }: { label: string; value: string; suffix: string; percent: number }) {
  return <div className="usage-item"><div><strong>{label}</strong><span><b>{value}</b>{suffix}</span></div><div className="usage-track"><i style={{ width: `${percent}%` }} /></div><small>{percent}% used</small></div>;
}

function AppRouter() {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}><Shell><Switch><Route path="/" component={Dashboard} /><Route path="/ingestion" component={Ingestion} /><Route path="/topology" component={Topology} /><Route path="/sentinel" component={GeoVlmSentinelModule} /><Route path="/changes" component={Changes} /><Route path="/field" component={Field} /><Route path="/exports" component={Exports} /><Route path="/billing" component={Billing} /><Route component={NotFound} /></Switch></Shell></ErrorBoundary>;
}

export default function App() {
  return <QueryClientProvider client={queryClient}><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><AppRouter /></WouterRouter><Toaster /></QueryClientProvider>;
}