"""Render a trace as a self-contained, offline HTML timeline."""

import json
from html import escape
from pathlib import Path
from typing import Any


def render_trace_html(events: list[dict[str, Any]], run_id: str, output: Path) -> Path:
    """Write an HTML viewer with embedded data and no external resources."""
    embedded = (
        json.dumps(events, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = _HTML.replace("__RUN_ID__", escape(run_id)).replace("__TRACE_DATA__", embedded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RepoPulse trace __RUN_ID__</title>
<style>
:root{color-scheme:dark;--bg:#0b1118;--panel:#121c27;--line:#273647;
--text:#e8f0f5;--muted:#9db0bf;--mint:#65e5c5;--amber:#ffc878;--red:#ff858c}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 80% 0%,#1b3440 0%,var(--bg) 35%);
color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.shell{max-width:1180px;margin:0 auto;padding:42px 26px 80px}
.eyebrow{color:var(--mint);font-size:12px;font-weight:800;letter-spacing:.18em;
text-transform:uppercase}
h1{font-size:clamp(32px,5vw,56px);line-height:1.1;margin:10px 0 12px;letter-spacing:-.04em}
.sub{color:var(--muted);max-width:800px;margin:0 0 28px}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:28px 0 30px}
.card{background:linear-gradient(150deg,#172633,var(--panel));border:1px solid var(--line);
border-radius:16px;padding:18px;min-height:105px}
.card strong{display:block;font-size:31px;line-height:1.2;color:var(--mint)}
.card span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 22px}
input,select{background:#14212c;border:1px solid #345063;border-radius:10px;
color:var(--text);padding:11px 13px;font:inherit}
input{min-width:300px;flex:1}
.count{color:var(--muted);font-size:13px}
.timeline{border-left:2px solid var(--line);margin-left:12px;padding-left:25px}
.event{position:relative;background:var(--panel);border:1px solid var(--line);
border-radius:15px;padding:17px 19px;margin:0 0 14px;box-shadow:0 8px 25px #0002}
.event:before{content:"";position:absolute;left:-33px;top:25px;width:12px;height:12px;
border-radius:50%;background:var(--mint);box-shadow:0 0 0 5px var(--bg)}
.event.warn:before{background:var(--amber)}.event.fail:before{background:var(--red)}
.eventhead{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.badge{display:inline-block;border:1px solid #3d736e;color:var(--mint);background:#173533;
border-radius:6px;padding:2px 8px;font-size:11px;font-weight:800;letter-spacing:.07em}
.warn .badge{color:var(--amber);border-color:#866b44;background:#382b20}
.fail .badge{color:var(--red);border-color:#854249;background:#382329}
time{color:var(--muted);font-size:12px}
.event h2{font-size:17px;margin:10px 0 4px;overflow-wrap:anywhere}
details{margin-top:10px}summary{color:var(--muted);cursor:pointer;font-size:13px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b151e;border:1px solid var(--line);
border-radius:10px;padding:13px;color:#c5d6e0;max-height:480px;overflow:auto;font-size:12px}
.footer{color:var(--muted);font-size:12px;margin-top:26px}
@media(max-width:720px){.cards{grid-template-columns:repeat(2,1fr)}.shell{padding:26px 16px}
input{min-width:100%}}
</style>
</head>
<body>
<main class="shell">
<div class="eyebrow">RepoPulse / local observability</div>
<h1>Agent run trace</h1>
<p class="sub">A chronological view of planning, tools, evidence, guardrails, memory,
usage, and the final answer. Run ID <span class="mono">__RUN_ID__</span>.</p>
<section class="cards" id="stats" aria-label="Run summary"></section>
<div class="toolbar">
<input id="search" type="search" placeholder="Filter events or evidence…"
aria-label="Filter trace events">
<select id="kind" aria-label="Event type"><option value="">All events</option></select>
<span class="count" id="count"></span>
</div>
<section class="timeline" id="timeline" aria-label="Trace events"></section>
<p class="footer">Generated locally from a redacted JSONL trace. No network assets or
scripts are loaded. Retrieved source text may still contain sensitive data.</p>
</main>
<script id="trace-data" type="application/json">__TRACE_DATA__</script>
<script>
const events=JSON.parse(document.getElementById('trace-data').textContent);
const $=id=>document.getElementById(id);
const tool=events.filter(e=>e.event==='tool_call');
const evidence=events.filter(e=>e.event==='evidence');
const blocked=events.filter(e=>e.event==='quarantine');
const final=events.findLast(e=>e.event==='final_answer');
const budget=final?.answer?.budget_units_used??0;
for(const [name,value] of [
  ['Events',events.length],['Tool attempts',tool.length],
  ['Evidence passages',evidence.length],['Quarantined',blocked.length]
]){
  const box=document.createElement('div');box.className='card';
  const strong=document.createElement('strong');strong.textContent=String(value);
  const span=document.createElement('span');span.textContent=name;
  box.append(strong,span);$('stats').append(box);
}
const budgetBox=document.createElement('div');budgetBox.className='card';
budgetBox.style.gridColumn='1 / -1';
const budgetText=document.createElement('span');
budgetText.textContent='Budget units used (character estimate or provider usage)';
const budgetValue=document.createElement('strong');budgetValue.textContent=String(budget);
budgetBox.append(budgetValue,budgetText);$('stats').append(budgetBox);
for(const type of [...new Set(events.map(e=>e.event))].sort()){
  const option=document.createElement('option');option.value=type;option.textContent=type;
  $('kind').append(option);
}
function title(e){
  if(e.event==='run_start')return e.question||'Run started';
  if(e.event==='plan')return (e.subquestions||[]).join(' · ')||'Plan';
  if(e.event==='guardrail')return `${e.guard} · ${e.decision}`;
  if(e.event==='llm_call')return `${e.stage} · ${e.status} · ${e.latency_ms} ms`;
  if(e.event==='tool_selection')return `${e.tool} · ${e.selected?'selected':'skipped'}`;
  if(e.event==='tool_call')return `${e.tool} · ${e.status} · ${e.latency_ms} ms`;
  if(e.event==='evidence')return `${e.kind||'source'} · ${e.title||e.url||''}`;
  if(e.event==='memory_read')return `${e.scope||'memory'} · ${e.records?.length||0} records`;
  if(e.event==='memory_write')return `${e.scope||'memory'} · ${e.status}`;
  if(e.event==='token_usage')return `${e.budget_units_used} budget units`;
  if(e.event==='final_answer')return e.answer?.summary?.text||'No cited summary';
  if(e.event==='quarantine')return `Excluded ${e.evidence_id}`;
  if(e.event==='run_end')return e.status||'Run complete';
  return e.reason||e.stage||e.action||e.event;
}
function draw(){
  const needle=$('search').value.toLocaleLowerCase();const type=$('kind').value;
  $('timeline').replaceChildren();let shown=0;
  for(const e of events){
    if(type&&e.event!==type)continue;
    if(needle&&!JSON.stringify(e).toLocaleLowerCase().includes(needle))continue;
    shown++;
    const card=document.createElement('article');card.className='event';
    if(['quarantine','degradation','validation_failure'].includes(e.event))
      card.classList.add('warn');
    if(e.status==='error'||e.event==='run_error'||e.event==='claim_rejection')
      card.classList.add('fail');
    const head=document.createElement('div');head.className='eventhead';
    const badge=document.createElement('span');badge.className='badge';
    badge.textContent=e.event.replaceAll('_',' ');
    const when=document.createElement('time');when.textContent=e.timestamp||'';
    head.append(badge,when);
    const h=document.createElement('h2');h.textContent=title(e);
    const details=document.createElement('details');
    const summary=document.createElement('summary');summary.textContent='Inspect event data';
    const pre=document.createElement('pre');pre.textContent=JSON.stringify(e,null,2);
    details.append(summary,pre);card.append(head,h,details);$('timeline').append(card);
  }
  $('count').textContent=`${shown} of ${events.length} events`;
}
$('search').addEventListener('input',draw);$('kind').addEventListener('change',draw);draw();
</script>
</body>
</html>
"""
