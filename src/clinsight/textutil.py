"""纯文本工具：不依赖任何第三方库，保证离线可跑、结果可复现。"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]")
_METRIC_RE = re.compile(r"\d+\s*(?:分钟|小时|min|h|%|％|例|次|人|天|周|月)")


def hits(text: str, cues: list[str]) -> int:
    """命中几个不同的线索词（去重，避免同一句话刷分）。"""
    return sum(1 for c in cues if c in text)


def matched(text: str, cues: list[str]) -> list[str]:
    return [c for c in cues if c in text]


def first_clause(text: str, limit: int = 60) -> str:
    """挑"最像痛点"的那一句当现象描述，而不是无脑取第一句。

    "随访也是个坑。出院时说好一个月复查……" 的第一句没有信息量，
    真正有证据价值的是后面带线索词的那句。
    """
    from .taxonomy import PAIN_CUES

    cues = [c for group in PAIN_CUES.values() for c in group]
    best, best_key = "", (0, 0)
    for part in _SENT_SPLIT.split(text):
        part = part.strip(" ，,、")
        if len(part) < 6:
            continue
        key = (sum(1 for c in cues if c in part), len(part))
        if key > best_key:
            best, best_key = part, key
    return (best or text.strip())[:limit]


def clip(text: str, limit: int = 90) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_metrics(text: str) -> list[str]:
    """抽出访谈里自述的量化线索，作为验收基线的候选。"""
    seen: list[str] = []
    for m in _METRIC_RE.findall(text):
        m = m.replace(" ", "")
        if m not in seen:
            seen.append(m)
    return seen


def detect_scene(text: str, fallback: str = "") -> str:
    from .taxonomy import SCENE_CUES, SCENE_DEFAULT

    best, best_score = "", 0
    for scene, cues in SCENE_CUES.items():
        s = hits(text, cues)
        if s > best_score:
            best, best_score = scene, s
    if not best:
        return fallback or SCENE_DEFAULT
    # 打平就回到本次访谈的主场景：一句话里顺带提到"床旁"，不该把需求挪到查房场景
    if fallback and fallback != best and hits(text, SCENE_CUES.get(fallback, [])) >= best_score:
        return fallback
    return best


def detect_frequency(text: str) -> tuple[str, int]:
    from .taxonomy import FREQUENCY_CUES, FREQUENCY_DEFAULT

    best = FREQUENCY_DEFAULT
    for cue, label, score in FREQUENCY_CUES:
        if cue in text and score > best[1]:
            best = (label, score)
    return best


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"
