from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from memrca.data.loader import Case
from memrca.propagation.dataset import ALREADY, CENSORED, N_BINS, STATIC_NAMES, build_incident
from memrca.propagation.onsets import anomaly_trace, detect_onsets, graph_distances, thresholds


def synthetic_case() -> Case:
    """db is the root (fault at 200 s); api, its caller, follows 40 s later; front stays normal."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 500)
    data = {f"{s}_{m}": 10 + rng.normal(0, 1, len(t)) for s in ("front", "api", "db") for m in ("cpu", "latency-90")}
    data["db_cpu"][t >= 202] += 60
    data["api_latency-90"][t >= 240] += 50
    return Case("demo", "db", "delay", 1, 200, pd.DataFrame(data, index=pd.Index(t, name="time")))


EDGES = pd.DataFrame({"source": ["front", "api"], "target": ["api", "db"]})  # caller -> callee


class OnsetTests(unittest.TestCase):
    def test_onsets_follow_the_injected_shifts(self) -> None:
        case = synthetic_case()
        trace = anomaly_trace(case, ["front", "api", "db"])
        onsets = detect_onsets(trace, thresholds(trace, int(trace.index.min()), 0), reference=0, horizon=300)
        self.assertLessEqual(abs(onsets["db"] - 2), 4)
        self.assertLessEqual(abs(onsets["api"] - 40), 4)
        self.assertTrue(np.isnan(onsets["front"]))

    def test_graph_distances_respect_call_direction(self) -> None:
        dist = graph_distances(EDGES, "db")
        self.assertEqual(dist.loc["api", "up"], 1)
        self.assertEqual(dist.loc["front", "up"], 2)
        self.assertTrue(np.isnan(dist["down"].get("front", np.nan)))
        self.assertEqual(dist.loc["front", "any"], 2)


class IncidentTests(unittest.TestCase):
    def test_labels_are_relative_to_the_root_onset(self) -> None:
        meta = {"case": "demo", "system": "x", "fault": "delay", "root": "db", "repetition": 1, "suite": "T"}
        inc = build_incident(synthetic_case(), meta, EDGES)
        api, db, front = (inc.services.index(s) for s in ("api", "db", "front"))
        self.assertEqual(inc.onset_bin[db], ALREADY)
        self.assertEqual(inc.onset_bin[front], CENSORED)
        self.assertAlmostEqual(inc.delay[api], 38, delta=5)  # api at ~40 s, root at ~2 s
        self.assertEqual(inc.dynamic.shape, (N_BINS, 3, inc.dynamic.shape[2]))
        self.assertEqual(inc.static[api, STATIC_NAMES.index("hop_up")] * 3, 1)
        self.assertEqual(inc.static[api, STATIC_NAMES.index("fault_delay")], 1)
        # by the time api's onset is confirmed, its callers see a callee that has been hit
        late = int(np.ceil((inc.delay[api] + 5) / 10))
        self.assertGreater(inc.dynamic[late, front, 3], 0)

    def test_unknown_fault_type_is_encoded(self) -> None:
        meta = {"case": "demo", "system": "x", "fault": "delay", "root": "db", "repetition": 1, "suite": "T"}
        inc = build_incident(synthetic_case(), meta, EDGES, fault_known=False)
        self.assertTrue((inc.static[:, STATIC_NAMES.index("fault_unknown")] == 1).all())
        self.assertTrue((inc.static[:, STATIC_NAMES.index("fault_delay")] == 0).all())


class ModelTests(unittest.TestCase):
    def test_forecast_is_a_cumulative_probability(self) -> None:
        import torch

        from memrca.propagation.model import PropagationNet, forecast

        torch.manual_seed(0)
        model = PropagationNet(len(STATIC_NAMES), 6)
        static = np.random.default_rng(1).random((4, len(STATIC_NAMES))).astype(np.float32)
        dynamic = np.random.default_rng(2).random((4, 6)).astype(np.float32)
        cum = forecast(model, static, dynamic, np.array([[0, 1], [1, 2]], dtype=np.int64), start_bin=3, n_bins=N_BINS,
                       elapsed_column=5)
        self.assertEqual(cum.shape, (N_BINS - 3, 4))
        self.assertTrue((np.diff(cum, axis=0) >= -1e-7).all())
        self.assertTrue(((cum >= 0) & (cum <= 1)).all())


if __name__ == "__main__":
    unittest.main()


class TelemetryTests(unittest.TestCase):
    def test_replay_health_matches_the_onsets_and_hides_the_answer(self) -> None:
        from memrca.propagation.telemetry import case_telemetry

        tele = case_telemetry(synthetic_case(), pre=150, post=290, step=2)
        services = tele["services"]
        ratio = dict(zip(services, tele["ratio"]))
        before = [t < 0 for t in tele["t"]]
        # the alert line comes from the normal period, so nothing crosses it before the fault
        self.assertTrue(all(v < 1 for s in services for v, b in zip(ratio[s], before) if b and v is not None))
        confirmed = dict(zip(services, tele["confirmed"]))
        self.assertLessEqual(abs(confirmed["db"] - 7), 4)       # onset ~2 s + 5 s persistence
        self.assertLessEqual(abs(confirmed["api"] - 45), 4)
        self.assertIsNone(confirmed["front"])
        self.assertEqual(tele["detect_s"], confirmed["db"])
        self.assertNotIn("root", tele)
        self.assertNotIn("fault", tele)


class ArchitectureTests(unittest.TestCase):
    def test_documented_links_only_join_services_that_exist(self) -> None:
        from memrca.propagation.architecture import documented_edges

        ss = ["front-end", "catalogue", "catalogue-db", "orders", "payment"]
        self.assertIn(("front-end", "catalogue"), documented_edges("ss", ss))
        self.assertIn(("orders", "payment"), documented_edges("ss", ss))
        self.assertTrue(all(a in ss and b in ss for a, b in documented_edges("ss", ss)))
        tt = ["ts-order-service", "ts-order-mongo", "ts-avatar-service", "ts-ghost-mysql"]
        self.assertEqual(documented_edges("tt", tt), [("ts-order-service", "ts-order-mongo")])
        self.assertEqual(documented_edges("xx", ss), [])
