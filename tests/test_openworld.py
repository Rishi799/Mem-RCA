from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from memrca.data.loader import Case
from memrca.openworld.console import ConsoleSession
from memrca.openworld.fingerprint import METRIC_TYPES, metric_shift_profile, pool
from memrca.openworld.incidents import cross_suite_incident_set, holdout_type_split, learned_root_logits
from memrca.openworld.memory import NOVEL, IncidentMemory
from memrca.openworld.metrics import novelty_auroc, oscr


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.centres = {"cpu": np.array([5.0, 0, 0, 0]), "mem": np.array([0, 5.0, 0, 0])}
        x, y = [], []
        for label, c in self.centres.items():
            x.append(c + rng.normal(0, 0.3, (20, 4)))
            y += [label] * 20
        self.x, self.y = np.vstack(x), y
        self.memory = IncidentMemory().fit(self.x, self.y)
        self.memory.calibrate(self.x, keep_fraction=0.95)

    def test_known_types_are_recognised(self) -> None:
        names, _ = self.memory.predict(self.centres["cpu"][None, :])
        self.assertEqual(names, ["cpu"])

    def test_unfamiliar_incident_is_novel(self) -> None:
        names, _ = self.memory.predict(np.array([[0, 0, 0, 5.0]]))
        self.assertEqual(names, [NOVEL])

    def test_enrolment_adds_a_type_without_retraining(self) -> None:
        disk = np.array([[0, 0, 5.0, 0]]) + np.random.default_rng(1).normal(0, 0.3, (3, 4))
        self.memory.enrol("disk", disk)
        names, _ = self.memory.predict(np.array([[0, 0, 5.0, 0]]))
        self.assertEqual(names, ["disk"])
        self.assertIn("cpu", self.memory.labels)

    def test_knn_memory_handles_a_type_with_two_modes(self) -> None:
        # "net" incidents look different depending on where they start; their mean sits between
        # the two modes, so a single prototype matches a real incident poorly, nearest neighbours well
        x = np.array([[5.0, 0, 0, 0], [5.1, 0, 0, 0], [0, 0, 5.0, 0], [0, 0, 5.1, 0], [0, 5.0, 0, 0], [0, 5.1, 0, 0]])
        y = ["net", "net", "net", "net", "mem", "mem"]
        probe = np.array([[5.05, 0, 0, 0]])
        proto = IncidentMemory().fit(x, y).similarities(probe)[0]
        knn = IncidentMemory("knn", k=1).fit(x, y).similarities(probe)[0]
        self.assertGreater(knn[1], proto[1])
        self.assertGreater(knn[1], 0.99)

    def test_split_memory_detects_with_knn_and_names_with_prototypes(self) -> None:
        memory = IncidentMemory("split", k=1).fit(self.x, self.y)
        probe = self.x[:1]
        np.testing.assert_allclose(memory.similarities(probe), self.memory.similarities(probe))
        self.assertAlmostEqual(float(memory.best_similarity(probe)[0]), 1.0)
        # an enrolled type gets its own examples for detection and a prototype for naming
        disk = np.array([[0, 0, 5.0, 0]]) + np.random.default_rng(1).normal(0, 0.3, (3, 4))
        memory.calibrate(self.x)
        memory.enrol("disk", disk)
        names, _ = memory.predict(disk[:1])
        self.assertEqual(names, ["disk"])


class ConsoleSessionTests(unittest.TestCase):
    def test_novel_incidents_are_grouped_and_enrolled(self) -> None:
        rng = np.random.default_rng(3)
        # metric-type shift pattern of each fault at the root; callers see a weaker copy
        patterns = {"cpu": [3, 0, 1, 0, 2, 0, 1, 0], "mem": [0, 3, 0, 1, 0, 2, 0, 1], "disk": [1, 1, 3, 3, 0, 0, 0, 2]}
        cases, faults, splits, statuses, shifts = [], [], [], [], []
        for fault, pattern in patterns.items():
            for split in ["train"] * 6 + ["validation"] * 6 + ["test"] * 6:
                unseen = fault == "disk"
                shifts.append(np.outer([1.0, 0.5, 0.2], pattern) + rng.normal(0, 0.1, (3, 8)))
                cases.append(f"{fault}_{len(cases)}")
                faults.append(fault)
                splits.append("test" if unseen else split)
                statuses.append("unseen" if unseen else "known")
        d = {"case": np.array(cases), "fault": np.array(faults), "root": np.array(["api"] * len(cases)),
             "services": np.array(["api", "db", "web"]), "shift": np.stack(shifts),
             "split": np.array(splits), "status": np.array(statuses)}
        stream = np.where(d["split"] == "test")[0]
        stream = stream[np.argsort(d["fault"][stream] == "disk", kind="stable")]  # known first, then disk
        session = ConsoleSession(d, "shift:anchor_shift+split1", stream)
        for _ in range(12):  # the known cpu and mem incidents
            session.next_incident()
        while not session.candidates:
            record = session.next_incident()
            self.assertEqual(record["verdict"], NOVEL)
        self.assertEqual(session.candidates[0]["truth_majority"], "disk")
        session.enrol(session.candidates[0]["id"], "disk-stress")
        record = session.next_incident()
        self.assertEqual(record["verdict"], "disk-stress")
        self.assertTrue(record["truth"]["correct"])
        self.assertEqual(session.state()["kpis"]["enrolled"], 1)


class IncidentSetTests(unittest.TestCase):
    def test_holdout_type_split_follows_the_protocol(self) -> None:
        fault = np.array(["cpu", "cpu", "cpu", "cpu", "mem", "mem"])
        repetition = np.array([1, 3, 4, 5, 1, 5])
        split, status = holdout_type_split(fault, repetition, unseen="mem")
        self.assertEqual(list(split), ["train", "validation", "test", "test", "test", "test"])
        self.assertEqual(list(status), ["known", "known", "known", "known", "unseen", "unseen"])

    def test_cross_suite_set_maps_new_incidents_onto_memory_services(self) -> None:
        memory = {"case": np.array(["a1", "a2", "a3"]), "fault": np.array(["cpu"] * 3), "root": np.array(["api"] * 3),
                  "services": np.array(["api", "db"]), "shift": np.ones((3, 2, 8)), "repetition": np.array([1, 2, 3])}
        new = {"case": np.array(["b1"]), "fault": np.array(["f1"]), "root": np.array(["db"]),
               "services": np.array(["db", "extra"]), "shift": np.full((1, 2, 8), 2.0)}
        d = cross_suite_incident_set(memory, new)
        self.assertEqual(list(d["split"]), ["train", "validation", "test", "test"])
        self.assertEqual(list(d["status"]), ["known", "known", "known", "unseen"])
        self.assertTrue(np.isnan(d["shift"][3, 0]).all())  # "api" is absent from the new suite
        self.assertTrue((d["shift"][3, 1] == 2.0).all())

    def test_learned_root_logits_fall_back_for_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            pd.DataFrame({"case": ["a"], "root_probabilities": [json.dumps({"api": 0.2, "db": 0.8})]}).to_csv(path, index=False)
            logits = learned_root_logits(path, np.array(["a", "b"]), ["api", "db"], fallback=np.array([0, 0]))
        self.assertEqual(int(logits[0].argmax()), 1)
        self.assertEqual(int(logits[1].argmax()), 0)


class MetricTests(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        known = np.array([0.9, 0.8, 0.95])
        unseen = np.array([0.1, 0.2])
        self.assertEqual(novelty_auroc(known, unseen), 1.0)
        # all knowns correct and separable -> OSCR is 1
        self.assertAlmostEqual(oscr(known, np.array([True, True, True]), unseen), 1.0)

    def test_oscr_is_bounded_by_closed_set_accuracy(self) -> None:
        known = np.array([0.9, 0.8, 0.95, 0.85])
        correct = np.array([True, False, True, True])
        self.assertAlmostEqual(oscr(known, correct, np.array([0.1, 0.2])), 0.75)


class FingerprintTests(unittest.TestCase):
    def test_anchor_keeps_root_row_and_weights_symptoms(self) -> None:
        features = np.array([[1.0, 0.0], [0.0, 4.0], [0.0, 2.0]])
        magnitude = np.array([1.0, 3.0, 1.0])
        out = pool(features, root=0, magnitude=magnitude, how="anchor")
        np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 3.5])
        np.testing.assert_allclose(pool(features, 0, magnitude, "global_mean"), [1 / 3, 2.0])
        np.testing.assert_allclose(pool(features, 0, magnitude, "hybrid"), [1.0, 0.0, 0.0, 3.5, 1.0, 4.0])

    def test_shift_profile_maps_metric_types(self) -> None:
        index = pd.Index(np.arange(0, 40), name="time")
        cpu = np.r_[np.full(20, 10.0) + np.random.default_rng(2).normal(0, 1, 20), np.full(20, 110.0)]
        metrics = pd.DataFrame({"api_cpu": cpu, "api_mem": np.full(40, 5.0)}, index=index)
        case = Case("demo", "api", "cpu", 1, 20, metrics)
        profile = metric_shift_profile(case, ["api"], horizon=20)
        self.assertGreater(profile[0, METRIC_TYPES.index("cpu")], 1.0)
        self.assertEqual(profile[0, METRIC_TYPES.index("mem")], 0.0)
        self.assertTrue(np.isnan(profile[0, METRIC_TYPES.index("diskio")]))


if __name__ == "__main__":
    unittest.main()
