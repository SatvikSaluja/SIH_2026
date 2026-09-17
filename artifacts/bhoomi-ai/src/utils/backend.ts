export async function api<T>(path:string, body?:unknown, signal?:AbortSignal):Promise<T> {
  const res=await fetch(`/api${path}`,{method:body===undefined?'GET':'POST',headers:body===undefined?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal});
  if(!res.ok) { const text=await res.text();let message=text;try{const data=JSON.parse(text);message=typeof data.detail==='string'?data.detail:data.error??text;}catch{}throw new Error(message||`HTTP ${res.status}`); }
  return res.json();
}
export const asset=(id:string,kind='image')=>`/api/workspace/jobs/${encodeURIComponent(id)}/artifact?kind=${kind}`;
