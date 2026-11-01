"""Create a self-contained browser replay for one recorded RCAEval incident.

The default score source is the Phase 1 Isolation Forest output. Pass a Phase 2
LSTM ``service_window_scores.csv`` with ``--scores`` after training to replay
the LSTM outputs through the same interface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from memrca.data.loader import load_case, metric_columns_by_service


SCORE_COLUMNS = [
    "fold", "system", "case", "window_start", "seconds_from_injection",
    "service", "anomaly_score",
]


def load_case_scores(path: Path, case: str, fold: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(SCORE_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Score CSV is missing required columns: {missing}")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=SCORE_COLUMNS, chunksize=250_000):
        match = chunk[(chunk["case"] == case) & (chunk["fold"] == fold)]
        if not match.empty:
            chunks.append(match)
    if not chunks:
        raise ValueError(f"No scores found for case={case!r}, fold={fold!r} in {path}")
    scores = pd.concat(chunks, ignore_index=True)
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna(subset=["anomaly_score"])
    return scores.sort_values(["service", "seconds_from_injection"]).reset_index(drop=True)


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_payload(case, scores: pd.DataFrame, edges: pd.DataFrame, max_points: int) -> dict:
    groups = metric_columns_by_service(case.metrics.columns)
    times = case.metrics.index.to_numpy(dtype=np.int64)
    stride = max(1, int(np.ceil(len(times) / max_points)))
    selected = np.arange(0, len(times), stride, dtype=int)
    if len(times) and selected[-1] != len(times) - 1:
        selected = np.append(selected, len(times) - 1)

    metric_series: dict[str, list[list[float | int | None]]] = {}
    for service, columns in groups.items():
        for column in columns:
            values = case.metrics[column].to_numpy()[selected]
            points = [
                [int(times[index] - case.inject_time), finite_number(value)]
                for index, value in zip(selected, values)
            ]
            metric_series[column] = points

    score_records = [
        {
            "service": str(row.service),
            "time": int(row.seconds_from_injection),
            "score": finite_number(row.anomaly_score),
        }
        for row in scores.itertuples(index=False)
    ]
    edge_records = [
        {"source": str(row.source), "target": str(row.target), "count": int(row.span_count)}
        for row in edges.itertuples(index=False)
    ]
    services = sorted(set(groups) | set(scores["service"].astype(str)) | {
        edge["source"] for edge in edge_records
    } | {edge["target"] for edge in edge_records})

    post = scores[(scores["seconds_from_injection"] >= 0) & (scores["seconds_from_injection"] <= 120)]
    ranking = []
    for service, group in post.groupby("service"):
        vals = group["anomaly_score"].to_numpy(dtype=float)
        if len(vals):
            top_count = max(1, int(np.ceil(len(vals) * 0.25)))
            ranking.append({"service": str(service), "score": float(np.sort(vals)[-top_count:].mean())})
    ranking.sort(key=lambda item: (-item["score"], item["service"]))

    return {
        "case": case.name,
        "system": case.name[3:5],
        "fault": case.fault,
        "groundTruthRoot": case.root_cause,
        "injectionTime": int(case.inject_time),
        "metricSeries": metric_series,
        "metricGroups": groups,
        "scores": score_records,
        "edges": edge_records,
        "services": services,
        "initialRanking": ranking[:10],
        "timeline": [int(times[index] - case.inject_time) for index in selected],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mem-RCA Incident Replay</title>
<style>
:root{--navy:#10264b;--blue:#2864b4;--teal:#18a6a6;--orange:#e77b24;--red:#d54d4d;--ink:#192638;--muted:#64748b;--line:#dce3ec;--paper:#f5f7fb;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 Segoe UI,Arial,sans-serif}header{background:var(--navy);color:#fff;padding:18px 26px;display:flex;align-items:center;justify-content:space-between;gap:12px}h1{font-size:21px;margin:0}header p{color:#d5e1f4;margin:3px 0 0}.badge{border:1px solid #6880a4;border-radius:999px;padding:5px 11px;color:#e9f2ff;font-size:12px;white-space:nowrap}
main{max-width:1450px;margin:auto;padding:18px}.summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}.pill,.card{background:var(--white);border:1px solid var(--line);border-radius:10px}.pill{padding:9px 13px;flex:1;min-width:150px}.pill small{display:block;color:var(--muted);text-transform:uppercase;font-size:10px;letter-spacing:.06em}.pill strong{font-size:14px}.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(360px,1fr);gap:14px}.card{padding:14px;min-width:0}.card h2{font-size:15px;margin:0 0 10px;color:var(--navy)}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}.controls button,select{border:1px solid var(--line);border-radius:7px;padding:7px 10px;background:white;color:var(--ink)}input[type=range]{width:min(600px,70%)}canvas{width:100%;height:290px;display:block}.sub{color:var(--muted);font-size:12px}.chart-label{font-size:12px;color:var(--muted);margin-top:4px}.graph{height:340px;width:100%;border:1px solid #e7ecf2;border-radius:8px;background:#fbfcfe}.rank{display:grid;grid-template-columns:24px minmax(100px,1fr) 2fr 60px;gap:8px;align-items:center;padding:7px 3px;border-bottom:1px solid #edf0f4;font-size:12px}.bar{height:8px;background:#e8edf5;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}.truth{color:#9c4a0b;font-weight:700}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin:6px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.warn{background:#fff8ed;border:1px solid #f2d5ab;padding:8px 10px;border-radius:8px;font-size:12px;margin:10px 0}.foot{margin-top:12px;font-size:11px;color:var(--muted)}@media(max-width:900px){.grid{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}.graph{height:300px}}
</style></head><body>
<header><div><h1>Mem-RCA | Recorded Incident Replay</h1><p id="subtitle"></p></div><span class="badge" id="modelBadge"></span></header>
<main><section class="summary" id="summary"></section>
<section class="card" style="margin-bottom:14px"><h2>Replay controls</h2><div class="controls"><button id="play">Play</button><button id="reset">Reset</button><label for="speed">Speed</label><select id="speed"><option value="1">1x</option><option value="3" selected>3x</option><option value="8">8x</option></select><input id="cursor" type="range" min="0" max="0" value="0"><span id="timeLabel" class="sub"></span></div><p class="sub">The dashed marker is the recorded fault injection time. Replay uses recorded telemetry and model/baseline scores.</p></section>
<div class="grid"><section class="card"><h2>Telemetry</h2><div class="controls"><label for="metric">Metric</label><select id="metric"></select><span class="sub" id="metricService"></span></div><canvas id="metricChart" width="900" height="330"></canvas><div class="chart-label">Seconds relative to injection | metric value</div></section>
<section class="card"><h2>Service dependency graph</h2><div class="legend"><span><i class="dot" style="background:#e77b24"></i>Top anomaly score</span><span><i class="dot" style="background:#2864b4"></i>Other service</span><span><i class="dot" style="background:#fff;border:2px solid #d54d4d"></i>Ground-truth root label</span></div><svg class="graph" id="graph" viewBox="0 0 640 340" role="img" aria-label="Observed service-call graph"></svg><p class="sub">Arrows show observed caller-to-callee trace edges. They are dependency evidence, not causal proof.</p></section>
<section class="card"><h2>Current service scores</h2><p class="sub">Nearest scored window to the replay cursor. Higher scores indicate greater rarity relative to that service's normal validation data.</p><div id="currentScores"></div></section>
<section class="card"><h2>Root-cause ranking</h2><p class="sub">Mean of each service's top-quartile scores during the first 120 seconds after injection.</p><div id="ranking"></div><div class="warn">The ground-truth root is shown separately for evaluation. This replay does not imply that trace edges establish causation or predict downstream impact time.</div></section></div>
<p class="foot">Mem-RCA research prototype | recorded RCAEval incident | no production telemetry connection</p></main>
<script>
const D=__DATA__;const MODEL=__MODEL__;document.title=`Mem-RCA replay - ${D.case}`;
document.getElementById('subtitle').textContent=`${D.case} · ${D.system.toUpperCase()} · ${D.fault} fault`;
document.getElementById('modelBadge').textContent=MODEL;
document.getElementById('summary').innerHTML='';
for(const [label,value] of [['System',D.system.toUpperCase()],['Fault type',D.fault],['Dataset root label',D.groundTruthRoot],['Telemetry points',D.timeline.length],['Observed graph edges',D.edges.length]]){let p=document.createElement('div');p.className='pill';let small=document.createElement('small');small.textContent=label;let strong=document.createElement('strong');strong.textContent=value;p.append(small,strong);document.getElementById('summary').append(p)}
const metricSelect=document.getElementById('metric');for(const key of Object.keys(D.metricSeries)){let o=document.createElement('option');o.value=key;o.textContent=key;metricSelect.append(o)}
const cursor=document.getElementById('cursor');cursor.max=Math.max(0,D.timeline.length-1);const chart=document.getElementById('metricChart'),ctx=chart.getContext('2d');
const byService={};for(const row of D.scores){(byService[row.service]??=[]).push(row)}for(const s of Object.keys(byService))byService[s].sort((a,b)=>a.time-b.time);
function nearestScore(service,t){const a=byService[service]||[];if(!a.length)return null;let best=a[0];for(const r of a){if(Math.abs(r.time-t)<Math.abs(best.time-t))best=r}return best}
function currentRows(t){return D.services.map(service=>({service,row:nearestScore(service,t)})).filter(x=>x.row&&x.row.score!==null).sort((a,b)=>b.row.score-a.row.score)}
function drawMetric(){let name=metricSelect.value;let points=D.metricSeries[name]||[];document.getElementById('metricService').textContent=name.split('_')[0];let W=chart.width,H=chart.height,L=58,R=16,T=16,B=34;ctx.clearRect(0,0,W,H);ctx.font='12px Segoe UI';ctx.fillStyle='#64748b';ctx.strokeStyle='#e5eaf1';ctx.lineWidth=1;
let vals=points.map(p=>p[1]).filter(v=>v!==null);if(!vals.length){ctx.fillText('No metric values available',L,40);return}let min=Math.min(...vals),max=Math.max(...vals);if(min===max){min-=1;max+=1}let xs=points.map(p=>p[0]);let xmin=Math.min(...xs),xmax=Math.max(...xs);if(xmin===xmax)xmax=xmin+1;
for(let i=0;i<=4;i++){let y=T+(H-T-B)*i/4;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();ctx.fillText((max-(max-min)*i/4).toPrecision(4),4,y+4)}
function x(v){return L+(v-xmin)/(xmax-xmin)*(W-L-R)}function y(v){return T+(max-v)/(max-min)*(H-T-B)}
ctx.strokeStyle='#2864b4';ctx.lineWidth=2;ctx.beginPath();let active=false;for(const p of points){if(p[1]===null){active=false;continue}if(!active){ctx.moveTo(x(p[0]),y(p[1]));active=true}else ctx.lineTo(x(p[0]),y(p[1]))}ctx.stroke();
if(xmin<=0&&xmax>=0){ctx.strokeStyle='#d54d4d';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(x(0),T);ctx.lineTo(x(0),H-B);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#a73b3b';ctx.fillText('injection',x(0)+4,T+12)}
let t=D.timeline[Number(cursor.value)]??0;ctx.strokeStyle='#e77b24';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(t),T);ctx.lineTo(x(t),H-B);ctx.stroke();ctx.fillStyle='#475569';ctx.fillText(`${xmin}s`,L,H-8);ctx.fillText(`${xmax}s`,W-R-42,H-8)}
function drawGraph(rows){const svg=document.getElementById('graph');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';const W=640,H=340,cx=W/2,cy=H/2,r=Math.min(130,50+8*D.services.length);const pos={};D.services.forEach((s,i)=>{const a=(-Math.PI/2)+2*Math.PI*i/Math.max(1,D.services.length);pos[s]=[cx+r*Math.cos(a),cy+r*Math.sin(a)]});let top=rows.length?rows[0].service:null;
let defs=document.createElementNS(ns,'defs');let marker=document.createElementNS(ns,'marker');marker.setAttribute('id','arrow');marker.setAttribute('markerWidth','8');marker.setAttribute('markerHeight','8');marker.setAttribute('refX','6');marker.setAttribute('refY','3');marker.setAttribute('orient','auto');let poly=document.createElementNS(ns,'polygon');poly.setAttribute('points','0 0, 6 3, 0 6');poly.setAttribute('fill','#8a9bb1');marker.append(poly);defs.append(marker);svg.append(defs);
for(const e of D.edges){let a=pos[e.source],b=pos[e.target];if(!a||!b)continue;let line=document.createElementNS(ns,'line');line.setAttribute('x1',a[0]);line.setAttribute('y1',a[1]);line.setAttribute('x2',b[0]);line.setAttribute('y2',b[1]);line.setAttribute('stroke','#9eacbd');line.setAttribute('stroke-width',String(Math.min(3,1+Math.log10(Math.max(1,e.count))*.25)));line.setAttribute('marker-end','url(#arrow)');svg.append(line)}
for(const s of D.services){const [x,y]=pos[s],g=document.createElementNS(ns,'g');let circle=document.createElementNS(ns,'circle');circle.setAttribute('cx',x);circle.setAttribute('cy',y);circle.setAttribute('r',26);circle.setAttribute('fill',s===top?'#f5a340':'#dce8f6');circle.setAttribute('stroke',s===D.groundTruthRoot?'#d54d4d':'#7892b5');circle.setAttribute('stroke-width',s===D.groundTruthRoot?'4':'1.5');let text=document.createElementNS(ns,'text');text.setAttribute('x',x);text.setAttribute('y',y+42);text.setAttribute('text-anchor','middle');text.setAttribute('font-size','11');text.setAttribute('fill','#26364b');text.textContent=s.length>20?s.slice(0,18)+'…':s;g.append(circle,text);svg.append(g)}if(!D.edges.length){let t=document.createElementNS(ns,'text');t.setAttribute('x',W/2);t.setAttribute('y',H-12);t.setAttribute('text-anchor','middle');t.setAttribute('font-size','12');t.setAttribute('fill','#64748b');t.textContent='No trace graph available for this incident';svg.append(t)}}
function drawScores(){let t=D.timeline[Number(cursor.value)]??0;document.getElementById('timeLabel').textContent=`${t>=0?'+':''}${t}s from injection`;const rows=currentRows(t);const root=document.getElementById('currentScores');root.replaceChildren();let max=Math.max(1,...rows.map(r=>r.row.score));for(const item of rows.slice(0,10)){let div=document.createElement('div');div.className='rank';let rank=document.createElement('b');rank.textContent=String(rows.indexOf(item)+1);let name=document.createElement('span');name.textContent=item.service;let track=document.createElement('span');track.className='bar';let fill=document.createElement('i');fill.style.width=`${Math.max(0,Math.min(100,item.row.score/max*100))}%`;track.append(fill);let score=document.createElement('span');score.textContent=item.row.score.toFixed(3);div.append(rank,name,track,score);root.append(div)}drawGraph(rows);drawMetric()}
const ranking=document.getElementById('ranking');for(let i=0;i<D.initialRanking.length;i++){const item=D.initialRanking[i],row=document.createElement('div');row.className='rank';const rank=document.createElement('b');rank.textContent=String(i+1);const name=document.createElement('span');name.textContent=item.service+(item.service===D.groundTruthRoot?' · root label':'');if(item.service===D.groundTruthRoot)name.className='truth';const bar=document.createElement('span');bar.className='bar';const fill=document.createElement('i');const max=Math.max(1,...D.initialRanking.map(x=>x.score));fill.style.width=`${Math.max(0,Math.min(100,item.score/max*100))}%`;bar.append(fill);const val=document.createElement('span');val.textContent=item.score.toFixed(3);row.append(rank,name,bar,val);ranking.append(row)}if(!D.initialRanking.length)ranking.textContent='No post-injection scores available for this case.';
cursor.addEventListener('input',drawScores);metricSelect.addEventListener('change',drawMetric);document.getElementById('reset').onclick=()=>{cursor.value=0;drawScores()};let timer=null;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';return}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{let next=Number(cursor.value)+Number(document.getElementById('speed').value);if(next>=Number(cursor.max)){cursor.value=cursor.max;clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}else cursor.value=next;drawScores()},100)};drawScores();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="RCAEval case folder name")
    parser.add_argument("--fold", default="fold1", choices=["fold1", "fold2", "fold3"])
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--scores", default="experiments/baseline/test_service_window_scores.csv")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--model-name", default="Phase 1 Isolation Forest")
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.max_points < 2:
        raise ValueError("--max-points must be at least 2")

    case = load_case(args.raw, args.case)
    scores = load_case_scores(Path(args.scores), args.case, args.fold)
    all_edges = pd.read_csv(args.edges)
    edges = all_edges[(all_edges["case"] == args.case) & (all_edges["system"] == args.case[3:5])]
    payload = build_payload(case, scores, edges, args.max_points)
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    document = HTML_TEMPLATE.replace("__DATA__", data_json).replace("__MODEL__", json.dumps(args.model_name))
    out_path = Path(args.out) if args.out else Path("experiments/demo") / f"{args.case}_{args.fold}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    print(f"Created replay: {out_path.resolve()}")
    print(f"Case: {case.name}; root-cause label: {case.root_cause}; graph edges: {len(edges)}")
    print(f"Scores: {len(scores)} rows from {args.model_name}")


if __name__ == "__main__":
    main()
