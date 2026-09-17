import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { MapContainer, GeoJSON, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { FeatureCollection, Polygon } from 'geojson';
import { api } from '../utils/backend';
import 'leaflet/dist/leaflet.css';
import type { Parcel } from '@workspace/api-client-react';

function Fit({data}: {data:FeatureCollection<Polygon>}) {
  const map=useMap();
  useEffect(()=>{ const bounds=L.geoJSON(data).getBounds(); if(bounds.isValid()) map.fitBounds(bounds,{padding:[20,20],maxZoom:19}); },[data,map]);
  return null;
}
export function ParcelMap({parcels}: {parcels:Parcel[]}) {
  if(!parcels.length) return <p>No parcel geometry is available for this ward.</p>;
  const local=parcels[0].coordinateSystem==='LOCAL_METRES';
  if(parcels.some(p=>p.coordinateSystem!==parcels[0].coordinateSystem))return <p>Select one coordinate system at a time.</p>;
  const data:FeatureCollection<Polygon>={type:'FeatureCollection',features:parcels.map(p=>({type:'Feature',id:p.id,geometry:p.geometry,properties:{id:p.id,area:p.areaSqM}}))};
  return <><p>{local?'Synthetic example · local coordinates in metres · no geographic location':'Georeferenced parcels · longitude/latitude; areas measured in the source projected CRS'}</p>
    <MapContainer key={`${local}-${parcels[0].regionId}`} crs={local?L.CRS.Simple:L.CRS.EPSG3857} center={[0,0]} zoom={0} minZoom={-8} style={{height:400,width:'100%'}}>
      {!local&&<TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="© OpenStreetMap contributors"/>}
      <GeoJSON key={JSON.stringify(data)} data={data} onEachFeature={(f,layer)=>layer.bindTooltip(`Face ${f.id} · ${Number(f.properties?.area).toFixed(2)} m²`)}/><Fit data={data}/>
    </MapContainer></>;
}
export function ParcelExplorerModal({isOpen,onClose,regionData}:{isOpen:boolean;onClose:()=>void;regionData:{name:string;code:string;bounds?:[[number,number],[number,number]]}}) {
  const result=useQuery({queryKey:['ward-parcels',regionData.code],queryFn:({signal})=>api<Parcel[]>(`/parcels?regionId=${encodeURIComponent(regionData.code)}`,undefined,signal),enabled:isOpen});
  useEffect(()=>{if(!isOpen)return;const close=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose();};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close);},[isOpen,onClose]);
  if(!isOpen)return null;
  return <div role="dialog" aria-modal="true" aria-label="Ward parcels" style={{position:'fixed',inset:24,zIndex:2000,background:'var(--surface)',padding:24,overflow:'auto',border:'1px solid var(--border)',boxShadow:'0 0 0 24px #0008'}}>
    <button className="button" onClick={onClose}>Close</button><h2>{regionData.name}</h2>
    {result.isPending?<p>Loading stored geometry…</p>:result.error?<p role="alert">{result.error.message}</p>:<><ParcelMap parcels={result.data}/><table className="data-table"><thead><tr><th>Internal parcel ID</th><th>Face</th><th>Area (m²)</th><th>Association</th></tr></thead><tbody>{result.data.map(p=><tr key={p.id}><td>{p.ulpin}</td><td>{p.id}</td><td>{p.areaSqM.toFixed(2)}</td><td>{p.status}</td></tr>)}</tbody></table></>}
  </div>;
}
