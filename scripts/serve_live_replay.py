"""Stream a recorded RCAEval incident to a browser dashboard over local SSE."""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, metric_columns_by_service  # noqa: E402


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mem-RCA live replay</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#17243a;font:14px Segoe UI,Arial}header{background:#10264b;color:white;padding:18px 24px}h1{font-size:21px;margin:0}header p{margin:5px 0 0;color:#cad8ed}main{max-width:1250px;margin:auto;padding:18px}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card,.stat{background:white;border:1px solid #dbe3ed;border-radius:10px;padding:14px}.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}.stat{flex:1;min-width:170px}.label{font-size:10px;color:#687991;text-transform:uppercase}.val{font-weight:650;margin-top:4px}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}button,select{border:1px solid #cbd5e1;border-radius:7px;padding:8px 12px;background:#fff;color:#17243a}button{cursor:pointer}button.primary{background:#2864b4;color:white;border-color:#2864b4}.muted{font-size:12px;color:#64748b}.graph{height:360px;width:100%;border:1px solid #e5eaf0;border-radius:8px;background:#fbfcfe}.item{display:grid;grid-template-columns:22px 1fr 2fr 60px;gap:8px;align-items:center;padding:8px 2px;border-bottom:1px solid #edf0f4;font-size:12px}.bar{height:8px;background:#e8edf5;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:#e77b24}.tag{font-size:11px;color:#25654e;background:#e8f6ef;padding:3px 7px;border-radius:99px}.log{max-height:190px;overflow:auto;font:12px Consolas,monospace}.warning{padding:9px;background:#fff8ed;border:1px solid #f0d8af;border-radius:8px;font-size:12px;margin-top:12px}.status{font-size:12px;color:#52667d} @media(max-width:850px){.row{grid-template-columns:1fr}}
</style></head><body><header><h1>Mem-RCA | Live Incident Replay</h1><p id="subtitle">Local stream of recorded telemetry and LSTM service scores</p></header><main>
<div class="stats" id="stats"></div><section class="card"><div class="controls"><button class="primary" id="play">Start stream</button><button id="reset">Reset</button><label>Replay speed <select id="speed"><option value="1">1x</option><option value="4" selected>4x</option><option value="10">10x</option></select></label><span class="status" id="status">Ready</span></div><div class="muted">Each update advances the recorded incident by one second. Persistent alerts require two consecutive scored windows above the validation threshold.</div></section>
<div class="row" style="margin-top:14px"><section class="card"><h2>Service graph and impact timing</h2><svg class="graph" id="graph" viewBox="0 0 700 360"></svg><div id="impacts"></div><p class="muted">A child alert delay is measured from its caller’s first persistent alert in this replay. Trace edges show observed calls, not causality.</p></section><section class="card"><h2>Current anomaly scores</h2><div class="muted" id="clock">Waiting for stream</div><div id="scores"></div><h2 style="margin-top:18px">Live telemetry snapshot</h2><div class="muted">One representative recorded metric per service, updated with the replay clock.</div><div id="telemetry"></div></section></div>
<section class="card" style="margin-top:14px"><h2>Alert event log</h2><div class="log" id="log"></div></section><div class="warning">Recorded dataset replay only. This prototype does not connect to production telemetry and does not predict future impact before the downstream alert appears.</div></main>
<script>const D=__DATA__;let stream=null,latest={},alarms={},seen={},lastId=-1;const $=x=>document.getElementById(x);$('subtitle').textContent=`${D.case} · ${D.system.toUpperCase()} · ${D.fault} fault · threshold ${D.threshold.toFixed(3)}`;
for(const [k,v] of [['Incident',D.case],['Root label',D.root],['Graph edges',D.edges.length],['Status','Ready']]){$('stats').insertAdjacentHTML('beforeend',`<div class="stat"><div class="label">${k}</div><div class="val">${v}</div></div>`)}
function graph(){const svg=$('graph'),ns='http://www.w3.org/2000/svg',W=700,H=360,cx=350,cy=165,r=Math.min(130,45+8*D.services.length),pos={};svg.replaceChildren();let defs=document.createElementNS(ns,'defs'),marker=document.createElementNS(ns,'marker'),arrow=document.createElementNS(ns,'polygon');marker.setAttribute('id','arrow');marker.setAttribute('markerWidth','8');marker.setAttribute('markerHeight','8');marker.setAttribute('refX','6');marker.setAttribute('refY','3');marker.setAttribute('orient','auto');arrow.setAttribute('points','0 0, 6 3, 0 6');arrow.setAttribute('fill','#8191a6');marker.append(arrow);defs.append(marker);svg.append(defs);D.services.forEach((s,i)=>{let a=-Math.PI/2+2*Math.PI*i/Math.max(1,D.services.length);pos[s]=[cx+r*Math.cos(a),cy+r*Math.sin(a)]});for(const e of D.edges){if(!pos[e.source]||!pos[e.target])continue;let l=document.createElementNS(ns,'line');l.setAttribute('x1',pos[e.source][0]);l.setAttribute('y1',pos[e.source][1]);l.setAttribute('x2',pos[e.target][0]);l.setAttribute('y2',pos[e.target][1]);l.setAttribute('stroke','#9aaabd');l.setAttribute('stroke-width','2');l.setAttribute('marker-end','url(#arrow)');svg.append(l)}for(const s of D.services){let [x,y]=pos[s],g=document.createElementNS(ns,'g'),c=document.createElementNS(ns,'circle'),t=document.createElementNS(ns,'text');c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',25);c.setAttribute('fill',alarms[s]?'#f5a340':'#dce8f6');c.setAttribute('stroke',s===D.root?'#d54d4d':'#7892b5');c.setAttribute('stroke-width',s===D.root?'4':'1.5');t.setAttribute('x',x);t.setAttribute('y',y+42);t.setAttribute('text-anchor','middle');t.setAttribute('font-size','11');t.textContent=s.length>18?s.slice(0,16)+'…':s;g.append(c,t);svg.append(g)}if(!D.edges.length){let t=document.createElementNS(ns,'text');t.setAttribute('x',350);t.setAttribute('y',340);t.setAttribute('text-anchor','middle');t.textContent='No trace graph available';svg.append(t)}}
function render(){const arr=Object.entries(latest).sort((a,b)=>b[1].score-a[1].score),max=Math.max(1,...arr.map(x=>x[1].score));$('scores').replaceChildren();for(const [i,[s,v]] of arr.entries()){let d=document.createElement('div');d.className='item';d.innerHTML=`<b>${i+1}</b><span>${s}${alarms[s]?' <span class="tag">ALERT</span>':''}</span><span class="bar"><i style="width:${Math.max(0,Math.min(100,100*v.score/max))}%"></i></span><span>${v.score.toFixed(3)}</span>`;$('scores').append(d)}graph();const imp=[];for(const e of D.edges){if(alarms[e.source]!=null&&alarms[e.target]!=null){let delay=alarms[e.target]-alarms[e.source];if(delay>=0)imp.push({edge:e,delay})}}$('impacts').innerHTML=imp.slice(0,8).map(x=>`<div class="item"><b>→</b><span>${x.edge.source} → ${x.edge.target}</span><span class="muted">observed alert delay</span><b>${x.delay}s</b></div>`).join('')||'<div class="muted">No ordered downstream alert pair yet.</div>';if(window.telemetryNow){$('telemetry').innerHTML=Object.entries(window.telemetryNow).map(([service,m])=>`<div class="item"><b>•</b><span>${service}</span><span class="muted">${m.name}</span><b>${m.value===null?'missing':Number(m.value).toPrecision(4)}</b></div>`).join('')}}
function event(ev){const x=JSON.parse(ev.data);$('clock').textContent=`Replay time ${x.time>=0?'+':''}${x.time}s relative to injection`;window.telemetryNow=x.telemetry||{};for(const r of x.rows){latest[r.service]=r;if(r.alarm&&!alarms[r.service]){alarms[r.service]=x.time;if(!seen[r.service]){$('log').insertAdjacentHTML('afterbegin',`<div>+${x.time}s persistent anomaly alert: ${r.service} (score ${r.score.toFixed(3)})</div>`);seen[r.service]=true}}}render()}
function connect(){if(stream){stream.close();stream=null;$('play').textContent='Resume stream';$('status').textContent='Paused';return}stream=new EventSource(`/events?speed=${$('speed').value}&from=${lastId}`);stream.onmessage=e=>{lastId=Number(e.lastEventId);event(e)};stream.addEventListener('done',()=>{stream.close();stream=null;$('play').textContent='Replay complete';$('status').textContent='Complete'});stream.onerror=()=>{$('status').textContent='Reconnecting…'}}
$('play').onclick=connect;$('reset').onclick=()=>{if(stream)stream.close();stream=null;lastId=-1;latest={};alarms={};seen={};$('log').replaceChildren();$('clock').textContent='Waiting for stream';$('play').textContent='Start stream';$('status').textContent='Ready';render()};graph();</script></body></html>'''


def make_payload(args: argparse.Namespace) -> dict:
    case = load_case(args.raw, args.case)
    path = Path(args.scores)
    score = pd.read_csv(path)
    require = {"fold", "system", "case", "split", "service", "target_time", "seconds_from_injection", "anomaly_score", "is_anomaly"}
    missing = require - set(score.columns)
    if missing:
        raise ValueError(f"Score CSV missing columns: {sorted(missing)}")
    score = score[(score.case == args.case) & (score.fold == args.fold)].copy()
    if score.empty:
        raise ValueError(f"No score rows for {args.case} in fold {args.fold}")
    validation = pd.read_csv(path)
    validation = validation[(validation.fold == args.fold) & (validation.system == args.case[3:5]) &
                            (validation.split == "validation") & (~validation.is_anomaly.astype(bool))]
    if validation.empty:
        raise ValueError("No normal validation scores available to set the replay alert threshold")
    grouped = validation.groupby(["case", "window_start"], as_index=False).anomaly_score.max()
    threshold = float(grouped.anomaly_score.quantile(.99))
    score = score.sort_values(["target_time", "service"])
    score["time"] = score.target_time.astype(int) - case.inject_time
    metrics = metric_columns_by_service(case.metrics.columns)
    edges_all = pd.read_csv(args.edges)
    edges_all = edges_all[(edges_all.system.astype(str) == args.case[3:5]) & (edges_all.case == args.case)]
    services = sorted(set(metrics) | set(score.service.astype(str)) | set(edges_all.source.astype(str)) | set(edges_all.target.astype(str)))
    rows_by_time: dict[int, list[dict]] = {}
    for row in score.itertuples(index=False):
        t = int(row.time)
        rows_by_time.setdefault(t, []).append({"service": str(row.service), "score": float(row.anomaly_score),
            "alarm": bool(row.anomaly_score > threshold)})
    times = sorted(set(int(x - case.inject_time) for x in case.metrics.index) | set(rows_by_time))
    metric_columns = {service: columns[0] for service, columns in metrics.items() if columns}
    metric_index = case.metrics.index.to_numpy(dtype=np.int64)
    telemetry_by_time: dict[int, dict] = {}
    for t in rows_by_time:
        absolute_time = case.inject_time + int(t)
        pos = int(np.searchsorted(metric_index, absolute_time))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(metric_index)]
        if candidates:
            row_index = min(candidates, key=lambda i: abs(int(metric_index[i]) - absolute_time))
            telemetry_by_time[t] = {}
            for service, column in metric_columns.items():
                value = case.metrics.iloc[row_index][column]
                numeric = float(value) if pd.notna(value) and np.isfinite(float(value)) else None
                telemetry_by_time[t][service] = {"name": column, "value": numeric}
    # Count consecutive alert windows per service in chronological order.
    consecutive: dict[str, int] = {}
    for t in sorted(rows_by_time):
        for row in rows_by_time[t]:
            service = row["service"]
            consecutive[service] = consecutive.get(service, 0) + 1 if row["alarm"] else 0
            row["alarm"] = consecutive[service] >= 2
    # Include a row only on score timestamps; browser keeps latest service score.
    return {"case": args.case, "system": args.case[3:5], "fault": case.fault, "root": case.root_cause,
            "threshold": threshold, "services": services,
            "edges": [{"source": str(r.source), "target": str(r.target), "count": int(r.span_count)} for r in edges_all.itertuples(index=False)],
            "steps": [{"time": t, "rows": rows_by_time.get(t, []), "telemetry": telemetry_by_time.get(t, {})} for t in sorted(rows_by_time)]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--fold", default="fold1", choices=["fold1", "fold2", "fold3"])
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--scores", required=True, help="Selected service_window_scores.csv for one fold/system")
    p.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    payload = make_payload(args)
    page = PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    steps = payload["steps"]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                body = page.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            if route.path == "/events":
                query = parse_qs(route.query)
                speed = max(.25, min(20.0, float(query.get("speed", ["4"])[0])))
                start = int(query.get("from", ["-1"])[0]) + 1
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.end_headers()
                try:
                    for i in range(start, len(steps)):
                        self.wfile.write(f"id: {i}\ndata: {json.dumps(steps[i], separators=(',', ':'))}\n\n".encode())
                        self.wfile.flush(); time.sleep(.35 / speed)
                    self.wfile.write(b"event: done\ndata: {}\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(404)

        def log_message(self, fmt, *args):
            print(f"[replay] {args[0]}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.case} on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
