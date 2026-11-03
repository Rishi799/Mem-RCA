"""Mem-RCA console: a local web app for the open-world incident memory.

    python scripts/serve_console.py            # then open http://127.0.0.1:8765

Scenarios replay recorded RCAEval incidents (no live telemetry), either one incident at a
time or as live monitoring that streams each recording's telemetry at a chosen speed:
  * RE2 folds - the memory knows four fault types; two more arrive unannounced;
  * RE1 folds - an independent suite; the localiser was retrained per fold;
  * RE2 memory -> RE3 - code-level faults the memory has never seen.
The Evidence tab summarises the experiment results written under experiments/.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.openworld.console import ConsoleSession  # noqa: E402
from memrca.propagation.service import PropagationForecaster  # noqa: E402
from memrca.propagation.telemetry import replay_telemetry  # noqa: E402
from memrca.openworld.incidents import (RE1_TYPES, cross_suite_incident_set, holdout_type_incident_set,  # noqa: E402
                                        load_profiles)

APP = PROJECT / "app"
SYSTEMS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


class Catalogue:
    """Available scenarios and the evidence summary, read from experiment outputs."""

    def __init__(self, experiments: Path, raw: Path) -> None:
        self.experiments, self.raw = experiments, raw
        configs = experiments / "open_world" / "results" / "selected_configs.json"
        self.re2_configs = json.loads(configs.read_text()) if configs.exists() else {}
        re3 = experiments / "external" / "results" / "re3_novel_types.csv"
        self.re3_configs = ({r.system: r.config for r in pd.read_csv(re3).itertuples() if r.role == "Mem-RCA"}
                            if re3.exists() else {})
        model = experiments / "propagation_forecast" / "propagation_net.pt"
        edges = experiments / "graph" / "re2_graph_all_edges.csv"
        self.forecaster = PropagationForecaster(model, edges, raw) if model.exists() and edges.exists() else None
        re1 = experiments / "external" / "results" / "re1_summary.csv"
        self.re1_configs = ({r.system: r.config for r in pd.read_csv(re1).itertuples()
                             if r.role == "Mem-RCA" and r.root == "pred"} if re1.exists() else {})
        self._telemetry: OrderedDict[str, dict] = OrderedDict()

    def telemetry(self, case: str) -> dict:
        if case not in self._telemetry:
            self._telemetry[case] = replay_telemetry(self.raw, case)
            while len(self._telemetry) > 16:
                self._telemetry.popitem(last=False)
        return self._telemetry[case]

    def _re1_predictions(self, system: str, unseen: str) -> Path:
        return (self.experiments / "external" / "re1" / "runs" / f"{system}_{unseen}" / "fold1" / system
                / "root_cause_case_predictions.csv")

    def scenarios(self) -> list[dict]:
        out = []
        for system, name in SYSTEMS.items():
            for fold in ("fold1", "fold2", "fold3"):
                path = self.experiments / "open_world" / "features" / "seed42" / fold / f"{system}.npz"
                if path.exists() and f"memrca_{system}" in self.re2_configs:
                    d = np.load(path, allow_pickle=False)
                    unseen = sorted(set(d["fault"][d["status"] == "unseen"]))
                    out.append({"id": f"re2:{system}:{fold}", "group": "RCAEval RE2 · resource faults",
                                "title": f"{name} · new types: {', '.join(unseen)}",
                                "description": f"Pattern library starts with {', '.join(sorted(set(d['fault'][d['split'] == 'train'])))}. "
                                               f"{' and '.join(unseen)} have never been seen and arrive unannounced.",
                                "config": self.re2_configs[f"memrca_{system}"]})
            for unseen in RE1_TYPES:
                if system in self.re1_configs and self._re1_predictions(system, unseen).exists():
                    out.append({"id": f"re1:{system}:{unseen}", "group": "RCAEval RE1 · independent recordings",
                                "title": f"{name} · new type: {unseen}",
                                "description": f"Independent recordings that were never used during development. The pattern library starts "
                                               f"with the other four resource faults; {unseen} has never been seen and "
                                               f"arrives unannounced.",
                                "config": self.re1_configs[system]})
            if system in self.re3_configs:
                out.append({"id": f"re3:{system}", "group": "RCAEval RE3 · code-level faults",
                            "title": f"{name} · code-level faults F1–F5",
                            "description": "The pattern library holds six resource fault types. Code-level faults (bugs in the "
                                           "service code) from a separate campaign arrive unannounced.",
                            "config": self.re3_configs[system]})
        return out

    def session(self, scenario_id: str) -> ConsoleSession:
        scenario = next(s for s in self.scenarios() if s["id"] == scenario_id)
        parts = scenario_id.split(":")
        if parts[0] == "re2":
            _, system, fold = parts
            d = dict(np.load(self.experiments / "open_world" / "features" / "seed42" / fold / f"{system}.npz",
                             allow_pickle=False))
        elif parts[0] == "re1":
            _, system, unseen = parts
            d = holdout_type_incident_set(load_profiles(self.experiments / "open_world" / "profiles", "re1", system),
                                          unseen, self._re1_predictions(system, unseen))
        else:
            system = parts[1]
            profiles = self.experiments / "open_world" / "profiles"
            d = cross_suite_incident_set(load_profiles(profiles, "re2", system), load_profiles(profiles, "re3", system))
        known = (d["split"] == "test") & (d["status"] == "known")
        unseen = (d["status"] == "unseen") & np.isin(d["split"], ["test", "test_extra"])
        stream = np.random.default_rng(7).permutation(np.where(known | unseen)[0])
        return ConsoleSession(d, scenario["config"], stream, self.raw, title=scenario["title"], system=system)

    def evidence(self) -> dict:
        e = self.experiments
        out: dict = {}

        def table(path: Path) -> pd.DataFrame | None:
            return pd.read_csv(path) if path.exists() else None

        runs = table(e / "open_world" / "results" / "open_set_by_run.csv")
        if runs is not None:
            shift = runs[runs["space"].eq("shift") & runs["pooling"].isin(["global_mean", "global_max", "anchor_pred"])]
            out["e2"] = shift.groupby(["system", "pooling"])[["auroc", "oscr"]].mean().round(3).reset_index().to_dict("records")
        selection = table(e / "open_world" / "results" / "selection_summary.csv")
        if selection is not None:
            out["re2_memrca"] = selection[selection["role"].isin(["Mem-RCA", "E2 anchor", "E2 global_max", "E2 global_mean"])][
                ["system", "role", "config", "auroc", "oscr", "closed_set_acc", "unseen_flagged_novel"]].round(3).to_dict("records")
        re1 = table(e / "external" / "results" / "re1_summary.csv")
        if re1 is not None:
            out["re1"] = re1[re1["root"].eq("pred")][["system", "role", "config", "auroc", "oscr", "closed_set_acc",
                                                       "unseen_flagged_novel"]].round(3).to_dict("records")
        re3 = table(e / "external" / "results" / "re3_novel_types.csv")
        if re3 is not None:
            out["re3"] = re3[["system", "role", "config", "auroc", "oscr", "unseen_flagged_novel", "known_accepted",
                              "mann_whitney_p"]].round(4).to_dict("records")
        roots = table(e / "open_world" / "results" / "root_by_case.csv")
        if roots is not None:
            out["shortcut"] = (roots.groupby(["system", "method"])["hit_at_1"].mean().round(3).reset_index()
                               .to_dict("records"))
        summary = PROJECT / "docs" / "figures" / "results_summary.json"  # written by scripts/make_figures.py
        if summary.exists():
            out["hypotheses"] = json.loads(summary.read_text(encoding="utf-8")).get("re1_hypotheses")
        for suite in ("re2", "re3"):
            prop = table(e / "propagation_forecast" / f"forecast_summary_{suite}.csv")
            if prop is not None:
                out[f"propagation_{suite}"] = prop[prop["start_s"].eq(0)][
                    ["split", "method", "system", "auroc", "c_index", "f1", "mae_s"]].round(3).to_dict("records")
            gain = table(e / "propagation_forecast" / f"forecast_update_gain_{suite}.csv")
            if gain is not None:
                out[f"propagation_gain_{suite}"] = gain.round(3).to_dict("records")
        holdout = table(e / "service_holdout" / "service_holdout_summary.csv")
        if holdout is not None:
            out["service_holdout"] = holdout[["system", "status", "method", "hit_at_1"]].round(3).to_dict("records")
        return out


def json_safe(value):
    """NaN/inf are not valid JSON; send them as null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_handler(catalogue: Catalogue, max_sessions: int = 8):
    """Each browser tab sends its own X-Session id, so tabs replay independently."""
    lock = threading.Lock()
    sessions: OrderedDict[str, ConsoleSession] = OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(json.dumps(json_safe(payload), allow_nan=False).encode(), "application/json", status)

        def _session_id(self) -> str:
            return self.headers.get("X-Session", "default")[:64]

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/scenarios":
                return self._json(catalogue.scenarios())
            if path == "/api/evidence":
                return self._json(catalogue.evidence())
            if path == "/api/state":
                session = sessions.get(self._session_id())
                return self._json(session.state() if session else None)
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            file = (APP / name).resolve()
            if APP.resolve() not in file.parents or not file.is_file():
                return self._send(b"not found", "text/plain", 404)
            self._send(file.read_bytes(), CONTENT_TYPES.get(file.suffix, "application/octet-stream"))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/live/peek":
                # telemetry of the recording the next incident comes from; its name (root, fault) stays on the server
                with lock:
                    session = sessions.get(self._session_id())
                    case = session.peek_case() if session else None
                    position, total = (session.position, len(session.stream)) if session else (0, 0)
                if session is None:
                    return self._json({"error": "start a scenario first"}, 400)
                if case is None:
                    return self._json({"done": True})
                return self._json({**catalogue.telemetry(case), "recording": position + 1, "total": total})
            try:
                with lock:
                    key = self._session_id()
                    if self.path == "/api/session":
                        sessions[key] = catalogue.session(body["scenario"])
                        sessions.move_to_end(key)
                        while len(sessions) > max_sessions:
                            sessions.popitem(last=False)
                        return self._json({"state": sessions[key].state()})
                    session = sessions.get(key)
                    if session is None:
                        return self._json({"error": "start a scenario first"}, 400)
                    if self.path == "/api/next":
                        incident = session.next_incident()
                        return self._json({"incident": incident, "state": session.state()})
                    if self.path == "/api/propagation":
                        context = session.propagation_context()
                        if catalogue.forecaster is None or context is None:
                            return self._json({"error": "propagation model not trained or no incident yet"}, 400)
                        forecast = catalogue.forecaster.incident(context["case"], context["system"], context["root"],
                                                                 context["fault"])
                        return self._json({**forecast, "verdict": context["verdict"]})
                    if self.path == "/api/enrol":
                        return self._json({"state": session.enrol(int(body["candidate"]), str(body.get("name", "")))})
            except (KeyError, StopIteration, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)
            self._json({"error": "unknown endpoint"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="experiments")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    catalogue = Catalogue(PROJECT / args.experiments, PROJECT / args.raw)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalogue))
    print(f"Mem-RCA console running at http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
