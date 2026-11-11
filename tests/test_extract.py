"""抽取层测试：保证"每条需求都能回到原话"这条契约不被破坏。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinsight import pipeline  # noqa: E402


class ExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        cls.points = __import__("clinsight.extract", fromlist=["x"]).extract_pain_points(cls.iv)
        cls.reqs = __import__("clinsight.extract", fromlist=["x"]).build_requirements(cls.points)

    def test_pain_points_found(self) -> None:
        self.assertGreaterEqual(len(self.points), 4)
        self.assertLessEqual(len(self.points), 8)

    def test_every_pain_has_quote_and_turn(self) -> None:
        for p in self.points:
            self.assertTrue(p.quote.strip(), f"{p.id} 没有原话")
            self.assertGreaterEqual(p.turn, 1)
            self.assertTrue(p.evidence, f"{p.id} 没有证据列表")

    def test_evidence_count_matches_evidence_list(self) -> None:
        for p in self.points:
            self.assertEqual(p.evidence_count, len(p.evidence))

    def test_severity_in_range(self) -> None:
        for p in self.points:
            self.assertGreaterEqual(p.severity, 1)
            self.assertLessEqual(p.severity, 5)

    def test_requirements_are_traceable(self) -> None:
        pain_ids = {p.id for p in self.points}
        self.assertEqual(len(self.reqs), len(self.points))
        for r in self.reqs:
            self.assertTrue(r.pain_ids)
            for pid in r.pain_ids:
                self.assertIn(pid, pain_ids, f"{r.id} 指向了不存在的痛点 {pid}")
            self.assertTrue(r.evidence, f"{r.id} 没有证据")
            self.assertGreaterEqual(len(r.acceptance), 3)

    def test_statement_is_not_boilerplate(self) -> None:
        # 需求陈述必须带上场景，不允许出现"通用"占位
        for r in self.reqs:
            self.assertIn("场景", r.statement)
            self.assertNotIn("{", r.statement)


if __name__ == "__main__":
    unittest.main()
