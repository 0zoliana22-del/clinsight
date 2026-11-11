"""MoSCoW 排序：把「拍脑袋定优先级」换成可复算、可答辩的加权模型。

三个刻意设计：
1. 每个因子都有独立贡献值，能回答"凭什么它是 Must"；
2. 合规风险是**扣分项**而不是否决项——大多数需求不是不能做，而是要付出合规成本；
3. 分数只用于排序，最终结论必须附一句人话解释（_decision_note）。
"""

from __future__ import annotations

from .models import PainPoint, Requirement

WEIGHTS: dict[str, float] = {
    "severity": 25.0,
    "frequency": 15.0,
    "reach": 10.0,
    "evidence": 15.0,
    "feasibility": 20.0,
    "cost": 10.0,
    "risk": 15.0,
}

THRESHOLDS: list[tuple[float, str]] = [
    (65.0, "Must"),
    (52.0, "Should"),
    (35.0, "Could"),
    (0.0, "Won't"),
]

LEVEL_TEXT = {"Must": "必须有", "Should": "应该有", "Could": "可以有", "Won't": "本期不做"}


def _moscow(score: float) -> str:
    for floor, label in THRESHOLDS:
        if score >= floor:
            return label
    return "Won't"


def _apply_floor(level: str, category: str, severity: int) -> str:
    """安全类需求不参与"降级"：涉及患者安全且严重度为 5 的，最低 Must。

    这条规则是刻意的产品治理设计——排序模型可以告诉你"性价比低"，
    但不能替你决定"这条命可以不管"。真要做取舍，也在 Must 内部砍范围，而不是降级。
    """
    if category == "safety" and severity >= 5:
        return "Must"
    return level


def apply(reqs: list[Requirement], points: list[PainPoint]) -> None:
    pmap = {p.id: p for p in points}
    for req in reqs:
        ps = [pmap[i] for i in req.pain_ids if i in pmap]
        if not ps:
            continue

        severity = max(p.severity for p in ps)
        freq_score = max(p.frequency_score for p in ps)
        freq_label = next(p.frequency for p in ps if p.frequency_score == freq_score)
        evidence_n = sum(p.evidence_count for p in ps)
        speakers = sorted({e["speaker"] for e in req.evidence})
        # 影响面不只看"谁在说"，还要看"谁被点到"：一条需求牵动医生、护士、患者三方，
        # 和只影响一个人，优先级不该一样。
        role_words = ("医生", "护士", "主任", "患者", "家属", "药师", "技师")
        stakeholders = {w for w in role_words for e in req.evidence if w in e["quote"]}
        people = sorted(set(speakers) | stakeholders)
        reach = max(1, len(people))

        feas = float(req.factors.get("ai_feasibility", 0.6))
        cost = float(req.factors.get("cost_efficiency", 0.7))
        risk = float(req.factors.get("compliance_risk", 0.1))

        contrib = {
            "severity": WEIGHTS["severity"] * severity / 5,
            "frequency": WEIGHTS["frequency"] * min(freq_score, 4) / 4,
            "reach": WEIGHTS["reach"] * min(reach, 3) / 3,
            "evidence": WEIGHTS["evidence"] * min(evidence_n, 5) / 5,
            "feasibility": WEIGHTS["feasibility"] * feas,
            "cost": WEIGHTS["cost"] * cost,
            "risk": -WEIGHTS["risk"] * risk,
        }
        score = round(max(0.0, min(100.0, sum(contrib.values()))), 1)
        raw_level = _moscow(score)
        req.score = score
        req.moscow = _apply_floor(raw_level, req.category, severity)
        if req.moscow != raw_level:
            level = f"{raw_level}（{req.moscow}，安全底线规则）"
        else:
            level = req.moscow
        req.factors.update(
            {
                "severity": severity,
                "frequency_score": float(freq_score),
                "reach": float(reach),
                "evidence_count": float(evidence_n),
                "compliance_risk": risk,
            }
        )
        req.rationale = [
            f"严重度 {severity}/5（{ps[0].impact}）→ +{contrib['severity']:.1f}",
            f"发生频次「{freq_label}」（{freq_score}/4）→ +{contrib['frequency']:.1f}",
            f"覆盖角色 {reach} 人（{'、'.join(people)}）→ +{contrib['reach']:.1f}",
            f"证据强度 {evidence_n} 条原话 → +{contrib['evidence']:.1f}",
            f"AI 可行性 {feas:.2f}（{req.ai_pattern}）→ +{contrib['feasibility']:.1f}",
            f"成本效率 {cost:.2f} → +{contrib['cost']:.1f}",
            f"合规风险 {risk:.2f}（{_flags_text(req.compliance_flags)}）→ {contrib['risk']:.1f}",
            f"合计 {score:.1f} → {level}（阈值 Must≥65 / Should≥52 / Could≥35）",
        ]
        req.rationale.append(_decision_note(req, feas, risk))


def _flags_text(flags: list[str]) -> str:
    names = {
        "clinical_advice": "输出临床建议",
        "phi": "涉及患者标识",
        "patient_facing": "面向患者",
        "decision_support": "辅助决策",
        "export": "数据导出",
    }
    return "、".join(names.get(f, f) for f in flags) if flags else "未触发"


def _decision_note(req: Requirement, feas: float, risk: float) -> str:
    if req.moscow == "Must":
        if req.category == "safety" and req.score < 65:
            return "结论：必须做（安全底线，不参与降级）。得分不高说明数据基础不足，落地先从人工核查清单开始，别硬上模型。"
        return "结论：先做。证据足、频次高、模型能力也够，2 周内能拿出可验证的试点结果。"
    if req.moscow == "Should":
        if risk >= 0.3:
            return "结论：第二梯队。价值成立，但合规成本会拉长工期，先做判定再排开发。"
        return "结论：第二梯队。价值明确但需要跨系统改造或更大样本验证，排在 Must 之后。"
    if req.moscow == "Could":
        if feas < 0.6:
            return "结论：观察。当前模型能力撑不住这个场景，先做人工流程 + 规则兜底，等技术或数据成熟。"
        return "结论：观察。需求真实但频次或证据不足，先记录不投入，下一轮访谈再验证。"
    return "结论：本期不做。要么超出能力边界，要么合规代价大于收益；写进「非目标」并说明替代方案。"


def summary(reqs: list[Requirement]) -> dict[str, object]:
    buckets: dict[str, list[str]] = {}
    for r in reqs:
        buckets.setdefault(r.moscow, []).append(r.id)
    order = ["Must", "Should", "Could", "Won't"]
    return {
        "counts": {k: len(buckets.get(k, [])) for k in order},
        "by_level": {k: buckets.get(k, []) for k in order},
        "top": sorted(reqs, key=lambda r: -r.score)[:3],
    }


# --------------------------------------------------------------------------
# 一致性校验：模型排出来的第一，和用户嘴上说的第一，是不是同一个？
#
# 这是我认为产品经理最该做、但很多工具不做的一件事：
# 打分模型永远会有偏差，偏差本身不可怕，可怕的是它悄悄覆盖了用户的原话。
# 不一致时不要自动调分——把冲突摆到桌面上，让人来裁决。
# --------------------------------------------------------------------------
PRIORITY_QUESTION_CUES = ("只能先", "最想先", "最想", "先解决哪个", "先解决什么", "先做一个", "选哪个")


def stated_priority(interview) -> dict[str, object] | None:
    for i, u in enumerate(interview.transcript[:-1]):
        if any(q in u.text for q in PRIORITY_QUESTION_CUES):
            nxt = interview.transcript[i + 1]
            if nxt.speaker != u.speaker:
                return {"turn": nxt.idx, "speaker": nxt.speaker, "text": nxt.text}
    return None


def _bigrams(s: str) -> set[str]:
    s = "".join(ch for ch in s if not ch.isspace())
    return {s[i : i + 2] for i in range(len(s) - 1)}


def check_alignment(reqs: list[Requirement], stated: dict[str, object] | None) -> dict[str, object]:
    if not stated or not reqs:
        return {"stated": stated, "status": "unknown", "note": "访谈里没有明确问「如果只能先做一个」，无法做一致性校验。"}

    text = str(stated["text"])
    grams = _bigrams(text)
    scored: list[tuple[int, Requirement]] = []
    for r in reqs:
        corpus = " ".join([r.scene, r.statement] + [e["quote"] for e in r.evidence])
        scored.append((len(grams & _bigrams(corpus)), r))
    scored.sort(key=lambda x: -x[0])
    best_overlap, matched = scored[0]
    top = max(reqs, key=lambda r: r.score)

    if best_overlap < 1:
        return {
            "stated": stated,
            "status": "unknown",
            "matched_id": None,
            "model_top_id": top.id,
            "note": "用户给了一个明确的首选，但没能映射到任何一条需求——需要当场追问一句「您说的具体是哪一步」。",
        }

    if matched.id == top.id:
        note = f"一致：用户原话落在 {matched.id}，模型排序第一也是它。这条可以直接写进 PRD 的第一版范围。"
        status = "aligned"
    else:
        note = (
            f"不一致：用户自述最想先解决的是 {matched.id}（{matched.scene} / "
            f"{_cat(matched.category)}），但模型按证据强度排出来第一是 {top.id}（{top.scene} / "
            f"{_cat(top.category)}，{top.score} 分）。不要自动改分——要么是用户的直觉被我们低估，"
            f"要么是这条需求的可解性其实撑不起它，两种都要人来判断。"
        )
        status = "conflict"

    return {
        "stated": stated,
        "status": status,
        "matched_id": matched.id,
        "model_top_id": top.id,
        "overlap": best_overlap,
        "note": note,
    }


def _cat(c: str) -> str:
    from .taxonomy import CATEGORY_LABEL

    return CATEGORY_LABEL.get(c, c)
