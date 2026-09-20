// LLM advisory layer over one ward's already-computed state (backend:
// geocadastra/api/advisory.py). Every number rendered here is a real
// aggregate the backend computed from stored rows; the model's only job is
// turning those into prose. When no provider is configured the backend
// returns facts with `brief: null` and says so -- this page shows that note
// verbatim rather than inventing a summary, and the same for /ask, which
// answers with a 503 the api() helper surfaces as its own detail string.
import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../utils/backend';
import { useWards, WardSelect } from './RealWorkspace';
import './real-workspace.css';

export interface WardBriefFacts {
  status: string; blocks_total: number; blocks_done: number; blocks_failed: number;
  conflicts_by_kind: Record<string, number>; total_conflicts: number;
  blocks_with_satisfied_area_constraints: number; blocks_with_area_reports: number;
}
export interface WardBrief { ward_job_id: number; facts: WardBriefFacts; brief: string | null; note?: string }
interface AskResult { ward_job_id: number; answer: string | null; tool_calls: { name: string }[]; error?: string }

export function Advisory() {
  const wards = useWards();
  const [wardId, setWard] = useState('');
  const [question, setQuestion] = useState('');
  const id = wardId ? Number(wardId.replace('ward-', '')) : null;

  const brief = useQuery({
    queryKey: ['advisory-brief', id],
    queryFn: ({ signal }) => api<WardBrief>(`/advisory/wards/${id}/brief`, undefined, signal),
    enabled: id !== null,
  });
  const ask = useMutation({
    mutationFn: (q: string) => api<AskResult>(`/advisory/wards/${id}/ask`, { question: q }),
  });

  const facts = brief.data?.facts;
  return <div className="real-workspace"><h1>Ward advisory</h1>
    <p>Plain-language state of one ward, and read-only questions answered against its stored rows. Nothing here edits geometry, resolves a conflict or reorders a survey.</p>
    {[wards.error, brief.error].filter(Boolean).map((e, i) => <p role="alert" key={i}>{e?.message}</p>)}

    <section><h2>Select a ward</h2>
      <WardSelect value={wardId} onChange={(v) => { setWard(v); ask.reset(); }} wards={wards.data?.wards ?? []} />
    </section>

    {id === null ? <p>Select a ward to load its recorded facts.</p>
      : brief.isPending ? <p>Loading recorded facts…</p>
      : facts && <>
        <section><h2>Recorded facts</h2>
          <section className="real-cards">
            <article><h3>Status</h3>{facts.status}</article>
            <article><h3>Blocks completed</h3>{facts.blocks_done}/{facts.blocks_total}</article>
            <article><h3>Blocks failed</h3>{facts.blocks_failed}</article>
            <article><h3>Conflicts recorded</h3>{facts.total_conflicts.toLocaleString('en-IN')}</article>
          </section>
          <p>Area constraints satisfied in {facts.blocks_with_satisfied_area_constraints} of {facts.blocks_with_area_reports} block(s) with an area report.</p>
          {Object.keys(facts.conflicts_by_kind).length
            ? <table><thead><tr><th>Conflict kind</th><th>Count</th></tr></thead>
                <tbody>{Object.entries(facts.conflicts_by_kind).sort((a, b) => b[1] - a[1]).map(([kind, count]) =>
                  <tr key={kind}><td>{kind}</td><td>{count.toLocaleString('en-IN')}</td></tr>)}</tbody></table>
            : <p>No fusion conflicts recorded for this ward.</p>}
        </section>

        <section><h2>Plain-language brief</h2>
          {brief.data?.brief
            ? <p>{brief.data.brief}</p>
            : <p role="status">{brief.data?.note ?? 'No brief was generated.'}</p>}
          <p>Every claim above is built only from the recorded facts in this page, so it can be checked against them directly.</p>
        </section>

        <section><h2>Ask about this ward</h2>
          <label>Question <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. which blocks failed, and what kind of conflicts dominate?" /></label>
          <button disabled={!question.trim() || ask.isPending} onClick={() => ask.mutate(question)}>
            {ask.isPending ? 'Asking…' : 'Ask'}</button>
          {ask.error && <p role="alert">{(ask.error as Error).message}</p>}
          {ask.data && <article>
            <p>{ask.data.answer ?? 'The advisory layer stopped before producing a final answer.'}</p>
            {ask.data.error && <p role="alert">{ask.data.error}</p>}
            <p>Backend data read to answer: {ask.data.tool_calls.length
              ? ask.data.tool_calls.map((t) => t.name).join(', ')
              : 'none — answered without reading further ward data'}</p>
          </article>}
        </section>
      </>}
  </div>;
}
