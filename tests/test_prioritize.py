"""排序层测试：分数可复算、安全底线不可降级、和用户直觉冲突要暴露。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinsight import pipeline, prioritize  # noqa: E402


class PrioritizeTest(unittest.TestCase):
    def test_score_is_reproducible(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        a = pipeline.analyze(iv)
        b = pipeline.analyze(iv)
        self.assertEqual([r.score for r in a.requirements], [r.score for r in b.requirements])
        self.assertEqual([r.moscow for r in a.requirements], [r.moscow for r in b.requirements])

    def test_moscow_matches_threshold(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        res = pipeline.analyze(iv)
        for r in res.requirements:
            if r.category == "safety" and r.factors.get("severity", 0) >= 5:
                self.assertEqual(r.moscow, "Must")
            else:
                self.assertEqual(r.moscow, prioritize._moscow(r.score))

    def test_safety_floor_never_downgraded(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-002.json")
        res = pipeline.analyze(iv)
        safety = [r for r in res.requirements if r.category == "safety"]
        self.assertTrue(safety, "INT-002 应该抽出交接安全隐患")
        for r in safety:
            self.assertEqual(r.moscow, "Must")
            # 底线项允许分数不高，但必须给出一句能说服人的解释
            self.assertTrue(any("安全底线" in line for line in r.rationale))

    def test_rationale_is_fully_attributed(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        res = pipeline.analyze(iv)
        for r in res.requirements:
            joined = "\n".join(r.rationale)
            for key in ("严重度", "发生频次", "覆盖角色", "证据强度", "AI 可行性", "合规风险", "合计"):
                self.assertIn(key, joined, f"{r.id} 缺少 {key} 的归因")

    def test_alignment_detects_conflict(self) -> None:
        # 护士长嘴上说"交接班"，模型却把"宣教"排到第一 —— 必须报冲突
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-002.json")
        res = pipeline.analyze(iv)
        self.assertEqual(res.priority_check["status"], "conflict")
        self.assertNotEqual(res.priority_check["matched_id"], res.priority_check["model_top_id"])

    def test_alignment_detects_agreement(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        res = pipeline.analyze(iv)
        self.assertEqual(res.priority_check["status"], "aligned")


if __name__ == "__main__":
    unittest.main()
