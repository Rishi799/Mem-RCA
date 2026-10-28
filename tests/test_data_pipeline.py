from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from memrca.baselines.isolation_forest import ServiceIsolationForest
from memrca.data.graph import service_edges_from_traces
from memrca.data.loader import Case
from memrca.data.windows import make_windows
from scripts.evaluate_graph_ranker import case_rankings


class WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.Index(np.arange(100, 110), name="time")
        metrics = pd.DataFrame({"api_cpu": np.arange(10, dtype=float)}, index=index)
        self.case = Case("demo", "api", "cpu", 1, 105, metrics)

    def test_windows_skip_injection_boundary_and_label_fault_period(self) -> None:
        result = make_windows(self.case, window_seconds=3, stride_seconds=2)
        self.assertEqual(result.metadata["window_start"].tolist(), [100, 102, 106])
        self.assertEqual(result.metadata["is_anomaly"].tolist(), [False, False, True])
        self.assertEqual(result.features["api_cpu"].tolist(), [1.0, 3.0, 7.0])

    def test_irregular_sampling_is_rejected(self) -> None:
        irregular = Case("bad", "api", "cpu", 1, 105,
                         self.case.metrics.iloc[[0, 1, 3, 4]].copy())
        with self.assertRaisesRegex(ValueError, "one-second metric sampling"):
            make_windows(irregular, window_seconds=2)


class GraphTests(unittest.TestCase):
    def test_deduplicates_spans_and_omits_same_service_edges(self) -> None:
        spans = pd.DataFrame([
            {"traceID": "t", "spanID": "1", "parentSpanID": "", "serviceName": "frontendservice"},
            {"traceID": "t", "spanID": "2", "parentSpanID": "1", "serviceName": "cartservice"},
            {"traceID": "t", "spanID": "2", "parentSpanID": "1", "serviceName": "cartservice"},
            {"traceID": "t", "spanID": "3", "parentSpanID": "2", "serviceName": "cartservice"},
            {"traceID": "t", "spanID": "4", "parentSpanID": "missing", "serviceName": "redis"},
        ])
        edges = service_edges_from_traces(spans, {"frontend", "cartservice", "redis"})
        self.assertEqual(edges.to_dict("records"), [
            {"source": "frontend", "target": "cartservice", "span_count": 1}
        ])


class IsolationForestTests(unittest.TestCase):
    def test_missing_values_fit_and_outlier_scores_higher(self) -> None:
        rng = np.random.default_rng(19)
        normal = pd.DataFrame(rng.normal(0, 0.3, size=(180, 3)), columns=["cpu", "mem", "latency"])
        normal.loc[::17, "mem"] = np.nan
        model = ServiceIsolationForest(random_state=17).fit(normal)
        ordinary_score = float(np.median(model.anomaly_score(normal.iloc[:20])))
        outlier = pd.DataFrame([[4.0, 5.0, 6.0]], columns=normal.columns)
        self.assertGreater(float(model.anomaly_score(outlier)[0]), ordinary_score)


class GraphRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = pd.DataFrame([
            {"fold": "fold1", "system": "ob", "case": "case1", "fault": "cpu",
             "fault_status": "known", "true_root": "frontend", "service": "frontend", "score": 0.5},
            {"fold": "fold1", "system": "ob", "case": "case1", "fault": "cpu",
             "fault_status": "known", "true_root": "frontend", "service": "checkout", "score": 1.5},
        ])

    def test_training_graph_child_support_can_rerank_upstream_candidate(self) -> None:
        ranked = case_rankings(self.scores, {"frontend": ["checkout"]}, alpha=1.0)
        root = ranked[ranked["service"].eq("frontend")].iloc[0]
        self.assertEqual(root["metric_rank"], 2)
        self.assertEqual(root["graph_rank"], 1)

    def test_missing_graph_falls_back_to_metric_only_ranking(self) -> None:
        ranked = case_rankings(self.scores, {}, alpha=2.0)
        self.assertTrue(np.array_equal(ranked["metric_rank"].to_numpy(), ranked["graph_rank"].to_numpy()))
        self.assertTrue(np.array_equal(ranked["metric_score"].to_numpy(), ranked["graph_score"].to_numpy()))


if __name__ == "__main__":
    unittest.main()
