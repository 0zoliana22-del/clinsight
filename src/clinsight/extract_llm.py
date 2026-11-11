"""v0.2：用 LLM 做需求抽取（规则版保留为兜底与对照）。

为什么只换"抽取"这一环：
- 排序要能当场复算、合规要能审计，这两环不能让模型碰；
- 抽取是纯语言理解任务，正好是模型相对规则的优势区；
- 单变量对照，才能说清楚"涨的分是哪来的"。

工程上的三个硬约束：
1. **证据必须逐字命中原文轮次**，否则整条丢弃——"没有原话就没有需求"这条契约不能因为换了模型就松；
2. **schema 校验 + 一次重试**，不合格就降级到规则版，绝不把脏数据流进下游；
3. **落盘缓存**，同一输入不重复计费，也让实验可复现。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import llm, taxonomy as T
from .extract import IMPACT_TEXT, _severity
from .models import Interview, PainPoint, Requirement
from .textutil import clip, detect_frequency, first_clause

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"
PROMPT_VERSION = "v2.0"

REQUIRED_CATEGORIES = set(T.PAIN_CUES)
SCENES = set(T.SCENE_CUES) | {"其他"}

SYSTEM = """你是资深医疗 AI 产品经理，正在做临床需求调研。
你的任务：从逐轮访谈记录中，抽出"可以独立排期、独立验收"的需求条目。

必须遵守：
1. 一条需求 = 一个可独立交付、可独立验收的能力。不要把一句话拆成两条，也不要把两件事塞进一条。
2. 每条需求必须绑定原文证据：turn 必须出现在给定记录里，quote 必须是该轮原话中的逐字片段（可以截取，但一个字都不能改）。
3. 需求陈述格式：「在『场景』中，为『角色』提供『能力』，让『可观测的结果』」。写目标，不写实现（不要出现"做一个按钮"）。
4. 不要把"访谈者的提问"当成证据；只引用受访者的话。
5. 抽 3~8 条。宁少勿多：证据不足就不要写，不要为凑数编造。
6. 只输出 JSON 对象，不要任何解释文字。"""

USER_TEMPLATE = """访谈信息：{role} / {department} / 主场景 {setting}

逐轮记录（JSON）：
{transcript}

请输出 JSON，格式：
{{
  "requirements": [
    {{
      "statement": "在「门诊」中，为心内科主治医师提供…，让…",
      "scene": "门诊",
      "category": "efficiency|toil|information|safety|communication|data",
      "evidence": [{{"turn": 7, "quote": "逐字原文片段"}}],
      "acceptance": "一句可证伪的验收信号（含指标与阈值）"
    }}
  ]
}}

scene 只能取：门诊 / 查房 / 病历文书 / 交接班 / 随访 / 急诊 / 科研数据 / 科室管理 / 患者宣教 / 其他
category 只能取：efficiency（效率损耗）/ toil（重复劳动）/ information（信息不可得）/ safety（安全与差错）/ communication（沟通与依从）/ data（数据与统计）"""


def _norm(text: str) -> str:
    return re.sub(r"[\s，。、；：！？,.!?;:'\"“”‘’（）()【】\[\]]", "", text)


def _input_hash(interview: Interview) -> str:
    payload = json.dumps(
        {
            "v": PROMPT_VERSION,
            "id": interview.id,
            "turns": [[u.idx, u.speaker, u.text] for u in interview.transcript],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _messages(interview: Interview) -> list[dict[str, str]]:
    transcript = json.dumps(
        [{"turn": u.idx, "speaker": u.speaker, "text": u.text} for u in interview.transcript],
        ensure_ascii=False,
        indent=1,
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                role=interview.role,
                department=interview.department,
                setting=interview.setting,
                transcript=transcript,
            ),
        },
    ]


def call_model(interview: Interview, use_cache: bool = True) -> dict:
    """调用模型（默认走缓存）。缓存键包含输入哈希与 prompt 版本，改了 prompt 自动失效。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _input_hash(interview)
    path = CACHE_DIR / f"{interview.id}-{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    data = llm.chat_json(_messages(interview), max_tokens=2048, timeout=90)
    path.write_text(
        json.dumps({"model": llm.config()["model"], "hash": key, "raw": data}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


def validate(data: dict, interview: Interview) -> tuple[list[dict], list[dict]]:
    """把模型输出校验成可信结构。任何一条证据对不上原文，整条需求丢弃。"""
    raw_items = data.get("requirements")
    if not isinstance(raw_items, list):
        return [], [{"reason": "顶层缺少 requirements 数组"}]

    turns = {u.idx: u for u in interview.transcript}
    good: list[dict] = []
    rejected: list[dict] = []

    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            rejected.append({"index": i, "reason": "不是对象"})
            continue
        statement = str(item.get("statement", "")).strip()
        category = str(item.get("category", "")).strip()
        scene = str(item.get("scene", "")).strip() or interview.setting
        acceptance = str(item.get("acceptance", "")).strip()

        if not (10 <= len(statement) <= 220):
            rejected.append({"index": i, "reason": f"statement 长度异常({len(statement)})"})
            continue
        if category not in REQUIRED_CATEGORIES:
            rejected.append({"index": i, "reason": f"category 非法: {category}"})
            continue
        if scene not in SCENES:
            scene = interview.setting

        evidence: list[dict] = []
        notes: list[str] = []
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            try:
                turn = int(ev.get("turn"))
            except (TypeError, ValueError):
                notes.append("turn 不是整数")
                continue
            quote = str(ev.get("quote", "")).strip()
            utterance = turns.get(turn)
            if utterance is None:
                notes.append(f"第 {turn} 轮不存在")
                continue
            if not quote:
                notes.append(f"第 {turn} 轮引用为空")
                continue
            if _norm(quote) in _norm(utterance.text):
                evidence.append({"turn": turn, "speaker": utterance.speaker, "quote": clip(quote)})
            else:
                notes.append(f"第 {turn} 轮找不到原文「{clip(quote, 24)}」")

        if not evidence:
            # 一条需求只记一条拒绝原因，避免"被拒条数"把一句话数成两条
            detail = f"（{'；'.join(notes[:3])}）" if notes else ""
            rejected.append({"index": i, "reason": f"没有一条证据能对上原文{detail}"})
            continue

        good.append(
            {
                "statement": statement,
                "scene": scene,
                "category": category,
                "evidence": evidence,
                "acceptance": acceptance,
            }
        )
    return good, rejected


def to_pain_points(items: list[dict], interview: Interview) -> list[PainPoint]:
    """把 LLM 结果映射成统一的中间层，好让下游（可行性/合规/排序）原样复用。"""
    points: list[PainPoint] = []
    for i, item in enumerate(items, start=1):
        turn = item["evidence"][0]["turn"]
        quote = item["evidence"][0]["quote"]
        freq_label, freq_score = detect_frequency(quote)
        points.append(
            PainPoint(
                id=f"PP-L{i:02d}",
                role=interview.role,
                scene=item["scene"],
                category=item["category"],
                quote=quote,
                turn=turn,
                phenomenon=first_clause(quote),
                frequency=freq_label,
                frequency_score=freq_score,
                severity=_severity(item["category"], quote),
                impact=IMPACT_TEXT.get(item["category"], "影响一线工作效率"),
                evidence_count=len(item["evidence"]),
                metrics=[],
                evidence=item["evidence"],
            )
        )
    return points


def to_requirements(items: list[dict], points: list[PainPoint]) -> list[Requirement]:
    reqs: list[Requirement] = []
    for i, (item, point) in enumerate(zip(items, points), start=1):
        acceptance = [item["acceptance"]] if item["acceptance"] else []
        acceptance += [
            f"{point.role}在真实「{point.scene}」中连续使用 5 个工作日，≥80% 的任务无需返工即可完成",
            "抽查 20 例：关键信息遗漏为 0；AI 生成内容 100% 带出处，且保留医生确认记录",
        ]
        reqs.append(
            Requirement(
                id=f"LLM-{i:03d}",
                statement=item["statement"],
                scene=item["scene"],
                category=item["category"],
                roles=[point.role],
                pain_ids=[point.id],
                evidence=list(item["evidence"]),
                acceptance=acceptance,
                ai_pattern="",
            )
        )
    return reqs


def merge(rule_reqs: list[Requirement], llm_reqs: list[Requirement]) -> list[Requirement]:
    """并集去重：同场景 + 同类别 + 有共同证据轮次，视为同一条需求。

    合并时保留 LLM 的措辞（更自然），但证据取并集（规则命中往往更全）。
    这不是最优解，是把"两个抽取器的分歧"变成"一条更完整的证据链"。
    """
    merged: list[Requirement] = list(rule_reqs)
    for cand in llm_reqs:
        turns = {e["turn"] for e in cand.evidence}
        hit = next(
            (
                r
                for r in merged
                if r.scene == cand.scene
                and r.category == cand.category
                and turns & {e["turn"] for e in r.evidence}
            ),
            None,
        )
        if hit is None:
            merged.append(cand)
            continue
        seen = {e["turn"] for e in hit.evidence}
        for ev in cand.evidence:
            if ev["turn"] not in seen:
                hit.evidence.append(ev)
                seen.add(ev["turn"])
        hit.pain_ids = list(dict.fromkeys(hit.pain_ids + cand.pain_ids))
        hit.statement = cand.statement
    return merged


def run(interview: Interview, use_cache: bool = True) -> tuple[list[PainPoint], list[Requirement], dict]:
    """返回 (痛点, 需求, 元信息)。调用方负责在异常时降级。"""
    data = call_model(interview, use_cache=use_cache)
    items, rejected = validate(data, interview)
    if not items:
        raise llm.LLMError(f"模型输出没有一条通过校验（{len(rejected)} 条被拒）")
    points = to_pain_points(items, interview)
    reqs = to_requirements(items, points)
    return points, reqs, {
        "model": llm.config()["model"],
        "accepted": len(items),
        "rejected": len(rejected),
        "rejections": rejected[:5],
    }
