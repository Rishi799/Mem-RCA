import unittest
import pandas as pd
from memrca.evaluation.inductive_ranker import (
    InductiveServiceRanker,
    extract_structural_node_features,
)
from memrca.evaluation.bootstrap_ci import bootstrap_metric_ci


class TestRobustnessEvaluation(unittest.TestCase):
    def setUp(self):
        self.services = ["frontend", "checkout", "payment", "email"]
        self.edges = pd.DataFrame([
            {"source": "frontend", "target": "checkout", "span_count": 10},
            {"source": "checkout", "target": "payment", "span_count": 5},
            {"source": "checkout", "target": "email", "span_count": 2},
        ])

    def test_structural_node_features(self):
        feats = extract_structural_node_features(self.services, self.edges)
        self.assertEqual(feats["frontend"]["out_degree"], 1.0)
        self.assertEqual(feats["frontend"]["in_degree"], 0.0)
        self.assertEqual(feats["frontend"]["is_entrypoint"], 1.0)
        self.assertEqual(feats["payment"]["is_leaf"], 1.0)

    def test_inductive_ranker_ordering(self):
        ranker = InductiveServiceRanker(graph_weight=0.2, topological_prior_weight=0.1)
        # Checkout has high score and calls payment & email
        scores = {"frontend": 1.0, "checkout": 4.5, "payment": 2.0, "email": 0.5}
        ranked = ranker.rank_case(scores, self.edges)
        self.assertEqual(ranked[0][0], "checkout")

    def test_bootstrap_metric_ci(self):
        ranks = [1, 1, 2, 1, 3, 1, 1, 2, 1, 1]
        ci = bootstrap_metric_ci(ranks, n_bootstraps=200, confidence_level=0.95)
        self.assertIn("Hit@1", ci)
        self.assertIn("Hit@3", ci)
        self.assertIn("MRR", ci)
        self.assertGreaterEqual(ci["Hit@1"][0], 0.6)
        self.assertLessEqual(ci["Hit@1"][1], ci["Hit@1"][0])
        self.assertGreaterEqual(ci["Hit@1"][2], ci["Hit@1"][0])


if __name__ == "__main__":
    unittest.main()
