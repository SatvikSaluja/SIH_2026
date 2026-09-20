import GeoVlmSentinelModule from "./components/GeoVlmSentinelModule";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState, useEffect, useRef } from 'react';
import { Link, Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import { IndiaBoundaryMap } from './components/IndiaBoundaryMap';
import { ParcelExplorerModal } from './components/ParcelExplorerModal';
import { TopologyModule as Topology } from './components/TopologyModule';
import { Dashboard, Ingestion, useWards, WardSelect } from './components/RealWorkspace';
import { Advisory } from './components/Advisory';
// Field verification reuses the operator console's panels rather than a
// third implementation -- they already drive the real endpoints.
import { ConflictsPanel } from './operator/components/ConflictsPanel';
import { EditPanel, conflictToPrefill } from './operator/components/EditPanel';
// The whole of the former standalone frontend/ app, folded in as one route:
// Overview, Datasets, Inference, Evidence, Review queue, Vision assistant,
// Training, and Ward operations (its own OperatorConsole). It is the only
// UI for /workspace/reviews and /workspace/vision, which nothing else here
// reaches, so deleting that app outright would have dropped real features.
import WorkspaceConsole from './operator/WorkspaceConsole';
import { ParcelMap } from './components/ParcelExplorerModal';
import {
  AlertTriangle,
  ArrowDownToLine,
  Brain,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Download,
  FileArchive,
  FlaskConical,
  Layers3,
  MapPinned,
  Menu,
  MoreHorizontal,
  Network,
  PackageCheck,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  UploadCloud,
  X,
} from 'lucide-react';
import {
  useCreateExport,
  useListChanges,
  useListRegions,
} from '@workspace/api-client-react';
import type {
  ChangeDetection,
  Region,
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
  { href: '/advisory', label: 'Ward advisory', icon: Sparkles },
  { href: '/operator', label: 'Operator console', icon: FlaskConical },
  { href: '/sentinel', label: 'Geo-VLM Sentinel', icon: Brain },
  { href: '/changes', label: 'Change detection', icon: Layers3 },
  { href: '/field', label: 'Field verification', icon: MapPinned },
  { href: '/exports', label: 'Export center', icon: ArrowDownToLine },
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
        {nav.slice(0, 8).map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} onClick={onClose} className={`nav-item ${location === href ? 'nav-active' : ''}`} data-testid={`link-${label.toLowerCase().replace(/\s+/g, '-')}`}>
            <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{href === '/' && <span className="nav-pulse" />}
          </Link>
        ))}
        <div className="nav-label nav-label-secondary">Governance</div>
        {nav.slice(8).map(({ href, label, icon: Icon }) => (
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

// Field verification, against the endpoint that actually performs it.
//
// This page used to drive PATCH /parcels/{id}, which the backend refuses
// with a 501 on purpose -- a parcel's status is not a field a client pokes;
// a surveyor's correction is a changeset move plus a durable SurveyPoint.
// So "Mark verified" could never have succeeded. It also filtered its queue
// on status 'pending', while a parcel is only ever 'matched' or 'unmatched',
// so the queue was permanently empty. Both halves were broken in a way that
// looked fine on screen.
//
// The real workflow needs a graph node id, and the only place a real one is
// handed out is a fusion conflict row (see EditPanel's own note). So the
// queue here is this ward's conflicts, and picking one prefills the move.
// Panels are the operator console's, not reimplemented -- which is why this
// renders inside .workspace-console, where their stylesheet is scoped.
function Field() {
  const wards = useWards();
  const [wardId, setWardId] = useState('');
  const [prefill, setPrefill] = useState<{ blockId: number; nodeId: number | null; x: number; y: number } | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const wardJobId = wardId ? Number(wardId.replace('ward-', '')) : null;

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">FIELD OPERATIONS / RECORDED EVIDENCE</div>
          <h1>Ground-truthing desk</h1>
          <p>Confirm a boundary corner from the field. Applied as a changeset move and recorded as a durable survey point, so Stage 6/7 calibration can use it like any other control point.</p>
        </div>
        <div className="heading-actions">
          <Button kind="secondary" onClick={() => { void wards.refetch(); setRefreshToken((t) => t + 1); }}>
            <RefreshCw size={15} /> Refresh
          </Button>
        </div>
      </div>

      <div className="workspace-console">
        <section className="panel">
          <h3>Ward</h3>
          <WardSelect value={wardId} onChange={(v) => { setWardId(v); setPrefill(null); }} wards={wards.data?.wards ?? []} />
          {wards.error && <p role="alert">{wards.error.message}</p>}
          <p className="muted">
            A corner can only be verified once every face touching it has a resolved parcel
            identity. On a freshly processed ward most corners do not yet, and the backend
            refuses them with “edit touches an unassigned face” — resolve identity for those
            parcels first (Operator console → Ward operations), then come back.
          </p>
        </section>

        {wardJobId === null ? (
          <p className="muted">Select a ward to load its recorded conflicts.</p>
        ) : (
          <>
            <ConflictsPanel
              wardJobId={wardJobId}
              refreshToken={refreshToken}
              onPickConflict={(c) => setPrefill(conflictToPrefill(c))}
            />
            <EditPanel
              wardJobId={wardJobId}
              prefill={prefill}
              onApplied={() => { setPrefill(null); setRefreshToken((t) => t + 1); }}
            />
          </>
        )}
      </div>
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

function AppRouter() {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}><Shell><Switch><Route path="/" component={Dashboard} /><Route path="/ingestion" component={Ingestion} /><Route path="/topology" component={Topology} /><Route path="/advisory" component={Advisory} /><Route path="/operator" component={WorkspaceConsole} /><Route path="/sentinel" component={GeoVlmSentinelModule} /><Route path="/changes" component={Changes} /><Route path="/field" component={Field} /><Route path="/exports" component={Exports} /><Route component={NotFound} /></Switch></Shell></ErrorBoundary>;
}

export default function App() {
  return <QueryClientProvider client={queryClient}><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><AppRouter /></WouterRouter><Toaster /></QueryClientProvider>;
}