# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU arithmetic checks for the restored final-metric helper.

Run directly with Python, or with unittest discovery. No GPU packages required.
"""

import copy
import json
import unittest

from _grand_mean_metrics import summarize_metric_samples


class GrandMeanMetricsTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(
            num_passes=2,
            num_rounds=3,
            num_warmup=20,
            drop_rounds=1,
            max_tokens=8,
            topk=2,
            ll=False,
        )
        # The first round of EACH pass is discarded. Rank 1 has different
        # counts, storage widths and timing; a ratio of means is not the mean
        # of these ratios. Combine's stored width and latency both double.
        self.rows = [
            [2, 1, 1000, 2000, 999, 999, 2, 4, 4, 8, 888, 888, 8, 16, 16, 32],
            [6, 2, 500, 1000, 777, 777, 1, 2, 2, 4, 666, 666, 4, 8, 8, 16],
        ]

    def test_mean_of_rank_and_round_ratios_and_actual_storage_widths(self):
        metrics, evidence = summarize_metric_samples(self.rows, **self.options)
        for phase in ("dispatch", "combine"):
            raw = evidence["metrics_unrounded"][phase]
            self.assertEqual(raw["avg_rdma_bandwidth_gbps"], 0.3515625)
            self.assertEqual(raw["avg_xgmi_bandwidth_gbps"], 0.9375)
            self.assertEqual(raw["avg_ll_bandwidth_gbps"], 5.0)
            self.assertEqual(metrics[phase]["bandwidth_gbps"], 0.35)
            self.assertEqual(metrics[phase]["bandwidth_metric"], "grand_mean")
        self.assertEqual(
            evidence["metrics_unrounded"]["dispatch"]["avg_latency_us"], 5.625
        )
        self.assertEqual(
            evidence["metrics_unrounded"]["combine"]["avg_latency_us"], 11.25
        )
        # All ranks have 1000 RDMA bytes for Dispatch; dividing by mean latency
        # would incorrectly produce 0.177777... instead of 0.3515625.
        self.assertNotAlmostEqual(1000 / (1000 * 5.625), 0.3515625)

    def test_drop_each_pass_and_preserve_all_raw_samples(self):
        original = copy.deepcopy(self.rows)
        metrics, evidence = summarize_metric_samples(self.rows, **self.options)
        self.assertEqual(self.rows, original)
        self.assertEqual(evidence["rank_rows"], original)
        self.assertEqual(evidence["retained_samples_per_rank_per_phase"], 4)
        self.assertEqual(evidence["retained_samples_per_phase"], 8)
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row[4:6] = [100000, 100000]
            row[10:12] = [200000, 200000]
        self.assertEqual(summarize_metric_samples(changed, **self.options)[0], metrics)
        self.assertEqual(json.loads(json.dumps(evidence)), evidence)

    def test_ll_uses_rank_zero_scale_after_global_mean(self):
        metrics, evidence = summarize_metric_samples(
            self.rows, **{**self.options, "ll": True}
        )
        self.assertEqual(evidence["ll_scale_rank0"], 16 / 3)
        self.assertEqual(metrics["dispatch"]["bandwidth_gbps"], 5.0)
        # Applying each rank's own LL scale before averaging is different.
        wrong = ((1.875 / 4) * (16 / 3) + (5.625 / 4) * (16 / 7)) / 2
        self.assertNotAlmostEqual(wrong, 5.0)

    def test_reject_invalid_sampling_parameters(self):
        for name, value in [
            ("num_passes", 0),
            ("num_rounds", 0),
            ("num_warmup", -1),
            ("drop_rounds", 3),
            ("drop_rounds", -1),
            ("topk", 0),
            ("max_tokens", -1),
            ("num_passes", 1.5),
            ("ll", 1),
        ]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                summarize_metric_samples(self.rows, **{**self.options, name: value})

    def test_reject_bad_rows_counts_widths_and_latencies(self):
        bad_rows = [[], [self.rows[0][:-1]], None]
        for column, value in [
            (0, -1),
            (1, 1.5),
            (2, 0),
            (3, float("inf")),
            (4, float("nan")),
            (6, 0),
            (7, -2),
            (8, True),
        ]:
            rows = copy.deepcopy(self.rows)
            rows[0][column] = value
            bad_rows.append(rows)
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_metric_samples(rows, **self.options)


if __name__ == "__main__":
    unittest.main()
