import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, asset } from '../utils/backend';
import { ParcelExplorerModal } from './ParcelExplorerModal';
import type { DashboardSummary } from '@workspace/api-client-react';
import './real-workspace.css';
export interface Ward {ward_job_id:number;source:string;status:string;n_blocks:number;completed_blocks:number;n_parcels:number;area_sqkm:number;synthetic:boolean;crs:string;created_at:string}
export const useWards=()=>useQuery({queryKey:['wards'],queryFn:({signal})=>api<{wards:Ward[]}>('/wards',undefined,signal),refetchInterval:4000});
export function WardSelect({value,onChange,wards}:{value:string;onChange:(id:string)=>void;wards:Ward[]}) {
  return <label>Ward <select value={value} onChange={e=>onChange(e.target.value)}><option value="">Select a ward</option>{wards.map(w=><option key={w.ward_job_id} value={`ward-${w.ward_job_id}`}>Ward {w.ward_job_id} · {w.source} · {w.status}</option>)}</select></label>;
}
const shown=(v:number|null|undefined)=>v==null?'Not measured':v.toLocaleString('en-IN');
export function Dashboard() {
  const wards=useWards(); const [selected,setSelected]=useState(''); const [open,setOpen]=useState(false);
  const summary=useQuery({queryKey:['real-dashboard'],queryFn:({signal})=>api<DashboardSummary>('/dashboard',undefined,signal),refetchInterval:4000});
  const data=summary.data; const current=wards.data?.wards.find(w=>`ward-${w.ward_job_id}`===selected);
  return <div className="real-workspace"><h1>Command center</h1><p>Stored ward jobs and measured outputs. Synthetic examples are identified by their source.</p>
    {(summary.error||wards.error)&&<p role="alert">{summary.error?.message??wards.error?.message}</p>}
    {summary.isPending?<p>Loading recorded results…</p>:data&&<><section className="real-cards"><article><h3>Completed ward area (km²)</h3>{shown(data.areaProcessedSqKm)}</article><article><h3>Stored parcel faces</h3>{shown(data.parcelsExtracted)}</article><article><h3>Survey accuracy</h3>{shown(data.accuracyScore)}</article><article><h3>Encroachments</h3>{shown(data.encroachments)}</article></section>
    <h2>Recorded activity</h2>{data.recentActivity.length?data.recentActivity.map(a=><p key={a.id}>{a.title} · {new Date(a.time).toLocaleString()} · {a.detail}</p>):<p>No ward activity recorded.</p>}</>}
    <h2>Inspect parcels</h2><WardSelect value={selected} onChange={v=>{setSelected(v);setOpen(false);}} wards={wards.data?.wards??[]}/><button disabled={!current} onClick={()=>setOpen(true)}>Open selected ward</button>
    {current&&<><p>{current.completed_blocks}/{current.n_blocks} blocks completed · {current.n_parcels} stored faces · {current.source}</p><ParcelExplorerModal isOpen={open} onClose={()=>setOpen(false)} regionData={{code:selected,name:`Ward ${current.ward_job_id} (${current.source})`}}/></>}
  </div>;
}
interface Catalog {datasets:{id:string;count:number;status:string;splits:Record<string,number>;limitations:string[]}[];checkpoints:{id:string;scope:string;bytes:number}[]}
interface Job {id:string;kind:string;status:string;dataset:string;tile?:string;checkpoint?:string;error?:string;elapsed_seconds?:number;checkpoint_sha256?:string;input_sha256?:string;evaluation_scope?:string;metrics?:Record<string,number|string>}
interface Training {id:string;scope:string;epochs:number;history:unknown[];result:unknown;checkpoints:string[]}
export function Ingestion() {
  const client=useQueryClient(); const wards=useWards(); const [wardId,setWard]=useState('');const [seed,setSeed]=useState(1);
  const [dataset,setDataset]=useState('');const [tile,setTile]=useState('');const [checkpoint,setCheckpoint]=useState('');const [epochs,setEpochs]=useState(1);const [selectedJob,setSelectedJob]=useState('');const [notice,setNotice]=useState('');
  const catalog=useQuery({queryKey:['workspace-catalog'],queryFn:({signal})=>api<Catalog>('/workspace/catalog',undefined,signal),refetchInterval:10000});
  const [offset,setOffset]=useState(0);
  const tiles=useQuery({queryKey:['workspace-tiles',dataset,offset],queryFn:({signal})=>api<{tiles:{id:string;split:string}[];total:number}>(`/workspace/tiles?dataset=${encodeURIComponent(dataset)}&offset=${offset}&limit=100`,undefined,signal),enabled:!!dataset});
  const jobs=useQuery({queryKey:['workspace-jobs'],queryFn:({signal})=>api<Job[]>('/workspace/jobs',undefined,signal),refetchInterval:3000});
  const training=useQuery({queryKey:['workspace-training'],queryFn:({signal})=>api<Training[]>('/workspace/training',undefined,signal),refetchInterval:10000});
  const action=useMutation({mutationFn:({path,body}:{path:string;body:unknown})=>api<Record<string,unknown>>(path,body),onSuccess:(data)=>{setNotice(`Backend accepted request${data.id?`: ${data.id}`:''}. Completion is shown in recorded job status.`);void client.invalidateQueries();},onError:()=>setNotice('')});
  const launch=(path:string,body:unknown)=>action.mutate({path,body});
  const selected=jobs.data?.find(j=>j.id===selectedJob);
  const currentWard=wards.data?.wards.find(w=>`ward-${w.ward_job_id}`===wardId);
  return <div className="real-workspace"><h1>Inference studio and Model Lab</h1><p>Run cataloged checkpoints on actual tiles. Training and inference status comes from the backend; no progress is inferred from elapsed time.</p>
    {[catalog.error,tiles.error,jobs.error,training.error,wards.error,action.error].filter(Boolean).map((e,i)=><p role="alert" key={i}>{e?.message}</p>)}{notice&&<p role="status">{notice}</p>}
    <section><h2>Dataset and checkpoint</h2><label>Dataset <select value={dataset} onChange={e=>{setDataset(e.target.value);setTile('');setOffset(0);}}><option value="">Select dataset</option>{catalog.data?.datasets.map(d=><option key={d.id} value={d.id}>{d.id} ({d.count} tiles)</option>)}</select></label>
    {!catalog.isPending&&!catalog.data?.datasets.length&&<p>No usable dataset is registered in this checkout. Add a validated dataset manifest to the backend catalog.</p>}
    <label>Tile <select value={tile} onChange={e=>setTile(e.target.value)}><option value="">Select tile</option>{tiles.data?.tiles.map(t=><option key={t.id} value={t.id}>{t.id} · {t.split}</option>)}</select></label>
    <button disabled={offset===0} onClick={()=>{setOffset(Math.max(0,offset-100));setTile('');}}>Previous tiles</button><button disabled={!tiles.data||offset+100>=tiles.data.total} onClick={()=>{setOffset(offset+100);setTile('');}}>Next tiles</button>
    <label>Checkpoint <select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)}><option value="">Select checkpoint</option>{catalog.data?.checkpoints.map(c=><option key={c.id} value={c.id}>{c.id}</option>)}</select></label>
    {!catalog.isPending&&!catalog.data?.checkpoints.length&&<p>No checkpoint is available. A model name alone is not a trained artifact.</p>}
    <p>{catalog.data?.checkpoints.find(c=>c.id===checkpoint)?.scope}</p>
    <button disabled={!dataset||!tile||!checkpoint||action.isPending} onClick={()=>launch('/workspace/inference',{dataset,tile,checkpoint,threshold_m:0.3})}>Run selected checkpoint</button>
    {dataset&&tile&&<img className="real-image" alt={`Input imagery for ${tile}`} src={`/api/workspace/tiles/${encodeURIComponent(dataset)}/${encodeURIComponent(tile)}/image`}/>}
    <h3>Train on selected dataset</h3><label>Epochs <input type="number" min={1} max={100} value={epochs} onChange={e=>setEpochs(Number(e.target.value))}/></label><button disabled={!dataset||!Number.isInteger(epochs)||epochs<1||epochs>100||action.isPending} onClick={()=>launch('/workspace/training',{dataset,epochs,batch_size:4})}>Start training</button>
    <p>Training requires validated RGB, measured nDSM, distance labels and valid masks. Unsupported inputs are rejected by the backend.</p></section>
    <section><h2>Recorded jobs and measured model comparisons</h2>{jobs.isPending?<p>Loading jobs…</p>:!jobs.data?.length?<p>No workspace jobs recorded.</p>:<table><thead><tr><th>Job/input</th><th>Checkpoint</th><th>Status</th><th>Measured IoU</th><th>Elapsed seconds</th></tr></thead><tbody>{jobs.data.map(j=><tr key={j.id}><td><button onClick={()=>setSelectedJob(j.id)}>{j.kind} · {j.dataset} / {j.tile??''} · {j.id.slice(0,8)}</button></td><td>{j.checkpoint??'Training'}</td><td>{j.status}{j.error&&<p role="alert">{j.error}</p>}</td><td>{j.metrics?.iou??'Not measured'}</td><td>{j.elapsed_seconds??'Not measured'}</td></tr>)}</tbody></table>}
    <p>Compare only matching inputs and evaluation settings. Tile reference agreement is not held-out generalization or survey accuracy.</p>
    {selected&&<article><h3>Job {selected.id}</h3><p>{selected.evaluation_scope}</p><p>Checkpoint SHA-256: {selected.checkpoint_sha256??'Not recorded yet'}</p><p>Input SHA-256: {selected.input_sha256??'Not recorded yet'}</p><pre>{JSON.stringify(selected.metrics??{},null,2)}</pre>{selected.status==='completed'&&selected.kind==='inference'&&<><img className="real-image" src={asset(selected.id)} alt="Stored model prediction"/><a href={asset(selected.id,'report')} download>Download recorded evaluation</a> · <a href={asset(selected.id,'arrays')} download>Download prediction arrays</a></>}</article>}
    <h3>Training history</h3>{training.data?.length?training.data.map(t=><details key={t.id}><summary>{t.id} · {t.epochs} recorded epochs · {t.scope}</summary><pre>{JSON.stringify({checkpoints:t.checkpoints,history:t.history,result:t.result},null,2)}</pre></details>):<p>No training history recorded.</p>}</section>
    <section><h2>Ward processing</h2><WardSelect value={wardId} onChange={setWard} wards={wards.data?.wards??[]}/><button disabled={!currentWard||action.isPending||currentWard.status==='running'} onClick={()=>launch('/processing/runs',{regionId:wardId,dataset:currentWard!.source})}>Run / resume selected ward</button>
    {currentWard&&<p>{currentWard.status} · {currentWard.completed_blocks}/{currentWard.n_blocks} blocks complete · {currentWard.source}</p>}
    <p>This processes the selected ward's stored inputs. It does not ingest the tile selected above. Real ward ingestion requires the raster/vector ingestion workflow; no file is substituted with a synthetic example.</p>
    <details><summary>Create an explicit synthetic example</summary><label>Seed <input type="number" value={seed} onChange={e=>setSeed(Number(e.target.value))}/></label><button disabled={!Number.isInteger(seed)||action.isPending} onClick={()=>launch('/processing/synthetic',{seed,width:50,height:35})}>Create synthetic ward</button><p>Generates local test geometry. Select the new ward to process it.</p></details></section>
  </div>;
}
