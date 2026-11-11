"""抽取层：口语访谈 → 痛点簇 → 可验证需求条目。

规则版的定位不是"效果天花板"，而是：
- 基线（baseline）：LLM 版本必须打赢它，否则不值得上模型；
- 兜底（fallback）：模型超时 / 无网 / 成本超标时仍有可用输出；
- 标注口径：规则怎么分，人标注就按同一套口径，评测才有意义。
"""

from __future__ import annotations

from . import taxonomy as T
from .models import Interview, PainPoint, Requirement
from .textutil import clip, detect_frequency, detect_scene, extract_metrics, first_clause, hits

IMPACT_TEXT: dict[str, str] = {
    "safety": "可能造成临床差错与医患纠纷，属于安全底线问题，不能只当体验优化",
    "efficiency": "直接转化为加班时长与可接诊量损失，是科室最敏感的成本项",
    "information": "信息获取成本高，医生只能凭记忆或反复问人，决策依据不稳定",
    "toil": "低价值重复劳动，挤压本应留给患者沟通的时间",
    "communication": "影响患者依从性与疗效，且每次都要重新解释一遍",
    "data": "取数依赖人工提需求，口径与时效都不可控，科研和质控都被卡住",
}


def _severity(category: str, text: str) -> int:
    base = T.CATEGORY_SEVERITY.get(category, 3)
    boost = 1 if hits(text, T.SEVERITY_BOOST_CUES) > 0 else 0
    return max(1, min(5, base + boost))


def _interview_frequency(interview: Interview) -> tuple[str, int]:
    joined = "".join(u.text for u in interview.transcript)
    return detect_frequency(joined)


def extract_pain_points(interview: Interview) -> list[PainPoint]:
    """按 (场景, 痛点类别) 聚类。同一件事被反复提，就自动累积证据强度。"""
    buckets: dict[tuple[str, str], list[dict]] = {}
    fallback_scene = interview.setting

    for u in interview.transcript:
        text = u.text.strip()
        if len(text) < 6:
            continue
        scores = {cat: hits(text, cues) for cat, cues in T.PAIN_CUES.items()}
        category, score = max(scores.items(), key=lambda kv: (kv[1], len(kv[0])))
        if score == 0:
            continue
        scene = detect_scene(text, fallback_scene)
        buckets.setdefault((scene, category), []).append(
            {
                "turn": u.idx,
                "speaker": u.speaker,
                "text": text,
                "score": score,
            }
        )

    if not buckets:
        return []

    # 频次优先取访谈里最常出现的说法，单句没写就用访谈整体
    freq_label, freq_score = _interview_frequency(interview)

    points: list[PainPoint] = []
    for i, ((scene, category), members) in enumerate(
        sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])), start=1
    ):
        members.sort(key=lambda m: (-m["score"], -len(m["text"])))
        head = members[0]
        local_freq = detect_frequency(head["text"])
        label, score = local_freq if local_freq[1] >= freq_score else (freq_label, freq_score)
        severity = max(_severity(category, m["text"]) for m in members)
        metrics: list[str] = []
        for m in members:
            for item in extract_metrics(m["text"]):
                if item not in metrics:
                    metrics.append(item)

        points.append(
            PainPoint(
                id=f"PP-{i:03d}",
                role=interview.role,
                scene=scene,
                category=category,
                quote=head["text"],
                turn=head["turn"],
                phenomenon=first_clause(head["text"]),
                frequency=label,
                frequency_score=score,
                severity=severity,
                impact=IMPACT_TEXT.get(category, "影响一线工作效率"),
                evidence_count=len(members),
                metrics=metrics,
                evidence=[
                    {"turn": m["turn"], "speaker": m["speaker"], "quote": clip(m["text"])} for m in members
                ],
            )
        )
    return points


def build_requirements(points: list[PainPoint]) -> list[Requirement]:
    """一个痛点簇对应一条需求，保证「需求 ↔ 证据」一一可溯。

    真实项目里会用 LLM 把多个痛点簇合并成一条需求（例如"门诊准备"合并
    效率+信息两类）。这里刻意保持 1:1，因为可解释性比合并更值钱。
    """
    reqs: list[Requirement] = []
    for i, p in enumerate(points, start=1):
        pb = T.SCENE_PLAYBOOK.get((p.scene, p.category), {})
        cap = str(pb.get("capability") or T.CAPABILITY_TEXT.get(p.category, "对应的辅助能力"))
        outcome = str(pb.get("outcome") or T.OUTCOME_TEXT.get(p.category, "目标指标改善"))
        metric = T.METRIC_NAME.get(p.category, "目标指标").format(scene=p.scene)
        direction = T.METRIC_DIRECTION.get(p.category, "下降")
        baseline = f"访谈自述 {'、'.join(p.metrics[:2])}" if p.metrics else "由科室埋点在试点前校准"

        acceptance = [
            f"试点 2 周，{metric}较基线{direction} ≥30%（基线：{baseline}）",
            f"{p.role}在真实「{p.scene}」中连续使用 5 个工作日，≥80% 的任务无需返工即可完成",
            "抽查 20 例：关键信息遗漏为 0；AI 生成内容 100% 带出处，且保留医生确认记录",
        ]
        if p.category == "safety":
            acceptance.insert(0, "漏报率优先于误报率：任何一次「该提示未提示」都按缺陷处理并回溯")
        if p.category in ("clinical",) or p.category == "communication":
            acceptance.append("对外话术发布前经护士长 / 医生确认，保留版本记录")

        reqs.append(
            Requirement(
                id=f"REQ-{i:03d}",
                statement=f"在「{p.scene}」场景，为{p.role}提供{cap}，让{outcome}。",
                scene=p.scene,
                category=p.category,
                roles=[p.role],
                pain_ids=[p.id],
                evidence=list(p.evidence),
                acceptance=acceptance,
                ai_pattern=str(T.FEASIBILITY_MAP.get(p.category, {}).get("pattern", "")),
                compliance_flags=[],
            )
        )
    return reqs
