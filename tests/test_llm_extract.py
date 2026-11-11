"""LLM 抽取层测试：不联网，用固定样本验证"校验"和"合并"这两段最容易出错的地方。

核心契约：**证据对不上原文，整条需求丢掉**。这条如果不测，线上就会出现
"需求说得很漂亮，但点开原文根本没有这句话"——那比抽漏了严重得多。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinsight import extract_llm, pipeline  # noqa: E402
from clinsight.models import Interview, Requirement, Utterance  # noqa: E402


def make_interview() -> Interview:
    return Interview(
        id="INT-T01",
        role="测试医生",
        department="测试科",
        setting="门诊",
        date="2026-01-01",
        duration_min=10,
        transcript=[
            Utterance(idx=1, speaker="访谈者", text="您好，聊聊门诊节奏。"),
            Utterance(idx=2, speaker="医生", text="我上午要看四十个病人，翻记录要翻半天。"),
            Utterance(idx=3, speaker="医生", text="病历还得手工写，一份份重复录入。"),
        ],
    )


class ValidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.iv = make_interview()

    def test_valid_item_passes(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供跨系统检索，让查资料从分钟级降到秒级。",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [{"turn": 2, "quote": "翻记录要翻半天"}],
                    "acceptance": "单次查资料耗时下降 30%",
                }
            ]
        }
        good, rejected = extract_llm.validate(data, self.iv)
        self.assertEqual(len(good), 1)
        self.assertEqual(rejected, [])

    def test_quote_not_in_source_is_dropped(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供某能力，让某指标改善。",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [{"turn": 2, "quote": "这句话医生根本没说过"}],
                    "acceptance": "",
                }
            ]
        }
        good, rejected = extract_llm.validate(data, self.iv)
        self.assertEqual(good, [])
        self.assertEqual(len(rejected), 1)

    def test_one_valid_evidence_saves_the_item(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供某能力，让某指标改善。",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [
                        {"turn": 2, "quote": "编造的引用"},
                        {"turn": 2, "quote": "翻记录要翻半天"},
                    ],
                    "acceptance": "",
                }
            ]
        }
        good, _ = extract_llm.validate(data, self.iv)
        self.assertEqual(len(good), 1)
        self.assertEqual(len(good[0]["evidence"]), 1)

    def test_bad_category_rejected(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供某能力，让某指标改善。",
                    "scene": "门诊",
                    "category": "vibes",
                    "evidence": [{"turn": 2, "quote": "翻记录要翻半天"}],
                }
            ]
        }
        good, rejected = extract_llm.validate(data, self.iv)
        self.assertEqual(good, [])
        self.assertIn("category", rejected[0]["reason"])

    def test_short_statement_rejected(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "优化一下",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [{"turn": 2, "quote": "翻记录要翻半天"}],
                }
            ]
        }
        good, _ = extract_llm.validate(data, self.iv)
        self.assertEqual(good, [])

    def test_unknown_turn_rejected(self) -> None:
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供某能力，让某指标改善。",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [{"turn": 99, "quote": "翻记录要翻半天"}],
                }
            ]
        }
        good, _ = extract_llm.validate(data, self.iv)
        self.assertEqual(good, [])

    def test_punctuation_noise_still_matches(self) -> None:
        # 模型经常顺手加标点，不该因此判成"对不上原文"
        data = {
            "requirements": [
                {
                    "statement": "在「门诊」中，为测试医生提供某能力，让某指标改善。",
                    "scene": "门诊",
                    "category": "information",
                    "evidence": [{"turn": 2, "quote": "翻记录，要翻半天。"}],
                }
            ]
        }
        good, _ = extract_llm.validate(data, self.iv)
        self.assertEqual(len(good), 1)


class MergeTest(unittest.TestCase):
    def _req(self, rid: str, scene: str, category: str, turns: list[int], statement: str) -> Requirement:
        return Requirement(
            id=rid,
            statement=statement,
            scene=scene,
            category=category,
            roles=["测试医生"],
            pain_ids=[],
            evidence=[{"turn": t, "speaker": "医生", "quote": f"第{t}轮原话"} for t in turns],
            acceptance=["a"],
            ai_pattern="",
        )

    def test_same_scene_category_and_turn_merges(self) -> None:
        rule = [self._req("REQ-001", "门诊", "information", [2, 5], "规则句")]
        llm = [self._req("LLM-001", "门诊", "information", [2], "模型句")]
        merged = extract_llm.merge(rule, llm)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].statement, "模型句")
        self.assertEqual({e["turn"] for e in merged[0].evidence}, {2, 5})

    def test_different_category_not_merged(self) -> None:
        rule = [self._req("REQ-001", "门诊", "information", [2], "规则句")]
        llm = [self._req("LLM-001", "门诊", "toil", [2], "模型句")]
        merged = extract_llm.merge(rule, llm)
        self.assertEqual(len(merged), 2)

    def test_no_shared_turn_not_merged(self) -> None:
        rule = [self._req("REQ-001", "门诊", "information", [2], "规则句")]
        llm = [self._req("LLM-001", "门诊", "information", [7], "模型句")]
        merged = extract_llm.merge(rule, llm)
        self.assertEqual(len(merged), 2)


class PipelineModeTest(unittest.TestCase):
    def test_llm_mode_without_key_degrades_to_rule(self) -> None:
        import os

        saved = os.environ.pop("CLINSIGHT_LLM_KEY", None)
        saved2 = os.environ.pop("OPENAI_API_KEY", None)
        try:
            iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / "INT-001.json")
            res = pipeline.analyze(iv, mode="llm", use_cache=False)
            self.assertTrue(res.requirements, "降级后必须仍有可用输出")
            self.assertTrue(res.stats.get("llm", {}).get("fallback"), "必须标记为降级，不能假装成功")
            self.assertIn("一页结论", res.prd_markdown)
        finally:
            if saved:
                os.environ["CLINSIGHT_LLM_KEY"] = saved
            if saved2:
                os.environ["OPENAI_API_KEY"] = saved2


if __name__ == "__main__":
    unittest.main()
