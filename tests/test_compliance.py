"""合规层测试：否定句、疑问句、风险等级，这几个点错了就是产品事故。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinsight import compliance, pipeline  # noqa: E402


class MentionTest(unittest.TestCase):
    def test_plain_hit(self) -> None:
        self.assertTrue(compliance.mentions("系统会自动给出诊断结论", "诊断"))

    def test_negated_hit_is_ignored(self) -> None:
        self.assertFalse(compliance.mentions("我们不是要 AI 下诊断，只是把原文摆出来", "诊断"))

    def test_interrogative_is_not_negation(self) -> None:
        # "算不算辅助诊断"是在问边界，不是否定，必须判为风险
        self.assertTrue(compliance.mentions("这个我们拿不准算不算辅助诊断", "诊断"))
        self.assertTrue(compliance.mentions("能不能把高危病人筛出来", "高危"))

    def test_negation_window_does_not_overreach(self) -> None:
        self.assertTrue(compliance.mentions("不应该漏掉，我们有诊断规范", "诊断"))


class InterviewComplianceTest(unittest.TestCase):
    def test_low_risk_interview_has_no_flags(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
        res = pipeline.analyze(iv)
        self.assertEqual(res.compliance["level"], "低")
        self.assertEqual(res.compliance["flags"], [])

    def test_decision_support_is_high_risk(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-003.json")
        res = pipeline.analyze(iv)
        self.assertEqual(res.compliance["level"], "高")
        self.assertIn("decision_support", res.compliance["flags"])
        self.assertIn("export", res.compliance["flags"])
        self.assertIn("暂不进入开发", str(res.compliance["gate"]))

    def test_high_risk_blocks_development(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-003.json")
        res = pipeline.analyze(iv)
        ids = {rl["id"] for rl in res.compliance["red_lines"]}
        self.assertIn("RL-03", ids)  # 器械属性判定
        self.assertIn("RL-04", ids)  # 可追溯（恒在）

    def test_checklist_is_complete(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-002.json")
        res = pipeline.analyze(iv)
        self.assertEqual(len(res.compliance["findings"]), 8)
        for f in res.compliance["findings"]:
            self.assertIn(f["status"], ("pass", "fail", "unknown"))

    def test_any_safety_requirement_raises_floor(self) -> None:
        iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-002.json")
        res = pipeline.analyze(iv)
        self.assertIn(res.compliance["level"], ("中", "高"))


if __name__ == "__main__":
    unittest.main()
