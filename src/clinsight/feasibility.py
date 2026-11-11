"""AI 可行性评估：判断这条需求落在模型能力边界内的哪个位置。

不做"能不能用 AI"的二元判断，而是给出：
- 推荐技术路径（Prompt / RAG / Agent / 规则 + 模型）
- 明确不能做什么（写进 PRD 的非目标）
- 必须保留的人审点（这决定合规能不能过）
- 上线前要看的评测指标
"""

from __future__ import annotations

from . import taxonomy as T
from .models import Requirement

DEFAULT_SPEC: dict[str, object] = {
    "score": 0.60,
    "pattern": "Prompt + 规则约束",
    "can": ["处理结构化程度中等的文本任务"],
    "cannot": ["替代人工判断"],
    "human_check": "输出需人工确认后使用",
    "eval": ["人工抽检通过率"],
}


def spec_for(req: Requirement) -> dict[str, object]:
    spec = dict(T.FEASIBILITY_MAP.get(req.category, DEFAULT_SPEC))
    pb = T.SCENE_PLAYBOOK.get((req.scene, req.category), {})
    if pb.get("pattern"):
        spec["pattern"] = pb["pattern"]
    if pb.get("feasibility"):
        spec["score"] = pb["feasibility"]
    return spec


def assess(reqs: list[Requirement]) -> dict[str, object]:
    rows = []
    for r in reqs:
        spec = spec_for(r)
        r.ai_pattern = str(spec["pattern"])
        r.factors["ai_feasibility"] = float(spec["score"])  # type: ignore[arg-type]
        r.factors["cost_efficiency"] = float(T.COST_EFFICIENCY.get(r.category, 0.6))
        rows.append(
            {
                "requirement_id": r.id,
                "scene": r.scene,
                "category": r.category,
                "pattern": spec["pattern"],
                "score": spec["score"],
                "can": spec["can"],
                "cannot": spec["cannot"],
                "human_check": spec["human_check"],
                "eval_metrics": spec["eval"],
            }
        )

    ranked = sorted(rows, key=lambda x: -float(x["score"]))  # type: ignore[arg-type]
    return {
        "rows": rows,
        "best": ranked[0] if ranked else None,
        "hardest": ranked[-1] if ranked else None,
        "summary": _summary(ranked),
    }


def _summary(ranked: list[dict[str, object]]) -> str:
    if not ranked:
        return "本次访谈没有抽出可评估的需求。"
    best, hardest = ranked[0], ranked[-1]
    boundary = str(hardest["cannot"][0]) if hardest["cannot"] else "能力边界受限"
    return (
        f"技术路径上最稳的是「{best['scene']}」场景（可行性 {best['score']}），建议作为第一版切口；"
        f"最吃力的是「{hardest['scene']}」场景（可行性 {hardest['score']}），"
        f"它的边界是「{boundary}」，必须配规则兜底 + 人审，别指望模型单点解决。"
    )
