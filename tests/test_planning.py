"""CPU invariants for cache planning; these are not NPU model parity tests."""

import copy
import json
from pathlib import Path
import unittest

from tools.planning import kv_estimate, visit_plan


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).resolve().parents[1] / "configs/model_shape.example.json").read_text())

    def test_visit_order_and_distinct_pass_owners(self):
        visits = visit_plan(self.config)
        self.assertEqual(len(visits), 48)
        self.assertEqual([(x["physical_layer"], x["pass"]) for x in visits[17:20]], [(17, 0), (6, 1), (7, 1)])
        self.assertEqual([(x["physical_layer"], x["pass"]) for x in visits[41:44]], [(17, 2), (18, 0), (19, 0)])
        self.assertEqual(len({x["kv_owner"] for x in visits}), 48)
        self.assertEqual(len({x["conv_owner"] for x in visits}), 48)
        self.assertEqual(sum(x["attention"] == "local" for x in visits), 40)

    def test_logical_kv_uses_sliding_window_and_full_context(self):
        # Analytic result: (40*1024 + 8*8192)*2*2*8*128 / 4.
        report = kv_estimate(self.config, tokens=8192, tp=4)
        self.assertEqual(report["local_kv_bytes_per_rank"], 41943040)
        self.assertEqual(report["global_kv_bytes_per_rank"], 67108864)
        self.assertEqual(report["kv_bytes_per_rank"], 109051904)
        doubled = kv_estimate(self.config, tokens=16384, tp=4)
        self.assertEqual(doubled["local_kv_bytes_per_rank"], report["local_kv_bytes_per_rank"])
        self.assertEqual(doubled["global_kv_bytes_per_rank"], 2 * report["global_kv_bytes_per_rank"])

    def test_multiple_requests_and_tp(self):
        a = kv_estimate(self.config, tokens=512, tp=2, requests=1)
        b = kv_estimate(self.config, tokens=512, tp=4, requests=3)
        self.assertEqual(b["kv_bytes_per_rank"] * 2, a["kv_bytes_per_rank"] * 3)
        self.assertEqual(a["global_owners"], 8)

    def test_head_dimension_not_hidden_divided_by_q(self):
        config = copy.deepcopy(self.config)
        config["text_config"]["hidden_size"] = 3072
        self.assertEqual(kv_estimate(config, tokens=2048), kv_estimate(self.config, tokens=2048))

    def test_rejects_invalid_tp_instead_of_rounding_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            kv_estimate(self.config, tokens=2048, tp=3)

    def test_rejects_variable_loop_and_invalid_boundaries(self):
        config = copy.deepcopy(self.config)
        config["text_config"]["loop_config"]["is_varloop"] = True
        with self.assertRaisesRegex(ValueError, "fixed-loop"):
            visit_plan(config)
        config["text_config"]["loop_config"]["is_varloop"] = False
        config["text_config"]["loop_config"]["core_end"] = 25
        with self.assertRaisesRegex(ValueError, "core range"):
            visit_plan(config)


if __name__ == "__main__":
    unittest.main()
