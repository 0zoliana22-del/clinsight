"""合规评估：在写 PRD 之前先判风险。

只做「风险提示 + 需补材料」，不做法学结论——这条边界要写进产品说明里，
因为一个需求洞察工具没有资格替代法规部门，越界就是给自己埋雷。

实现上有个小但重要的细节：否定句不算命中。「不要 AI 下诊断」这句话里
出现了「诊断」，但它恰恰是在划红线，不能当成风险信号。
"""

from __future__ import annotations

from . import taxonomy as T
from .models import Interview, Requirement
from .textutil import hits

RISK_WEIGHTS: dict[str, float] = {
    "clinical_advice": 0.35,
    "decision_support": 0.25,
    "phi": 0.15,
    "patient_facing": 0.12,
    "export": 0.10,
}

FLAG_NAMES: dict[str, str] = {
    "clinical_advice": "输出临床建议",
    "phi": "涉及患者标识信息",
    "patient_facing": "面向患者交付",
    "decision_support": "辅助诊疗决策",
    "export": "数据导出 / 外发",
}

NEGATORS = ("不是", "不要", "不用", "不能", "不做", "不会", "无需", "别", "禁止", "不准")
# 疑问句里的「能不能 / 算不算」不是在否定，是用户在问边界——这类句子恰恰是风险信号
INTERROGATIVES = ("能不能", "算不算", "是不是", "可不可以", "要不要", "是否", "会不会")


def mentions(text: str, cue: str, window: int = 8) -> bool:
    """命中线索词，但跳过被否定的表述。"""
    start = 0
    while True:
        i = text.find(cue, start)
        if i < 0:
            return False
        prefix = text[max(0, i - window) : i]
        if any(q in prefix for q in INTERROGATIVES):
            return True
        if not any(n in prefix for n in NEGATORS):
            return True
        start = i + len(cue)


def assess_requirement(req: Requirement, interview: Interview) -> tuple[list[str], float]:
    text = " ".join([req.statement] + [e["quote"] for e in req.evidence])
    flags = [f for f, cues in T.COMPLIANCE_CUES.items() if any(mentions(text, c) for c in cues)]
    risk = 0.05 + sum(RISK_WEIGHTS.get(f, 0.0) for f in flags)
    if req.category == "safety":
        risk += 0.05
    risk = round(min(risk, 0.95), 2)
    req.compliance_flags = flags
    req.factors["compliance_risk"] = risk
    return flags, risk


def assess_interview(interview: Interview, reqs: list[Requirement]) -> dict[str, object]:
    flags: list[str] = []
    for r in reqs:
        for f in r.compliance_flags:
            if f not in flags:
                flags.append(f)

    all_text = " ".join(u.text for u in interview.transcript) + " " + interview.notes
    # 访谈里说出口、但最终没进需求的东西同样要判风险：
    # "能不能筛出高危病人"没被排进本期需求，不代表这句话不构成医疗器械信号。
    for f, cues in T.COMPLIANCE_CUES.items():
        if f not in flags and any(mentions(all_text, c) for c in cues):
            flags.append(f)

    mentions_research = any(k in all_text for k in ("科研", "论文", "发表", "课题"))

    findings = [
        _finding(
            "CK-01",
            "数据处理合法性基础",
            "unknown",
            "本次为模拟访谈数据，不含真实患者信息；落地前需确认院内数据使用授权与患者知情同意的覆盖范围。",
            "《个人信息保护法》第 13 条：处理敏感个人信息需取得单独同意",
            "向医院信息科 / 伦理办公室索取数据使用授权模板",
        ),
        _finding(
            "CK-02",
            "脱敏与最小必要",
            "fail" if "phi" in flags else "unknown",
            "需求文本中出现患者标识相关表述，入模前必须按字段清单脱敏。"
            if "phi" in flags
            else "访谈本身未见显式患者标识，但真实接入需先定义 PHI 字段清单与脱敏规则。",
            "《个人信息保护法》敏感个人信息条款；数据最小必要原则",
            "输出 PHI 字段清单 + 脱敏方案，作为开发准入条件",
        ),
        _finding(
            "CK-03",
            "部署形态",
            "unknown",
            "部署形态未定。涉及患者数据时优先私有化 / 专有云，禁止把原始病历直接送公网 API。",
            "《数据安全法》；医院网络安全等级保护要求",
            "在技术方案中明确模型部署位置与数据流向图",
        ),
        _finding(
            "CK-04",
            "医疗器械属性判定",
            "fail" if ("decision_support" in flags or "clinical_advice" in flags) else "pass",
            "检测到诊疗决策相关表述，可能落入医疗器械管理范围，需先做属性判定。"
            if ("decision_support" in flags or "clinical_advice" in flags)
            else "当前需求被判定为效率 / 信息类工具，不构成诊疗决策；但结论需法规同事书面留档。",
            "《人工智能医用软件产品分类界定指导原则》",
            "与法规 / 注册同事开一次 30 分钟界定会，结论写入 PRD 附录",
        ),
        _finding(
            "CK-05",
            "人审与留痕",
            "pass",
            "全部需求的方案设计都保留了人工确认点（医生核对引用 / 护士确认话术 / 关键节点不可跳过）。",
            "《生成式人工智能服务管理暂行办法》；电子病历应用管理规范",
            "把「确认记录」字段写进埋点，确保可回放",
        ),
        _finding(
            "CK-06",
            "输出边界说明",
            "fail" if "patient_facing" in flags else "unknown",
            "存在面向患者的输出，必须在材料上明示不确定性、覆盖范围与「不替代医生」的定位。"
            if "patient_facing" in flags
            else "界面尚未设计边界说明文案，需在交互稿固定位置声明覆盖范围与不作为诊断依据。",
            "产品责任；《生成式人工智能服务管理暂行办法》标识义务",
            "在原型中预留固定免责区，文案经法务确认",
        ),
        _finding(
            "CK-07",
            "伦理审查",
            "unknown",
            "涉及科研 / 发表 / 对照实验时必须先过伦理，即便只是效率对比。"
            if mentions_research
            else "本次未涉及科研用途；若后续用于论文或对外发布，需补伦理审查。",
            "《涉及人的生命科学和医学研究伦理审查办法》",
            "确认用途边界，科研用途单独走伦理流程",
        ),
        _finding(
            "CK-08",
            "不良事件与回滚",
            "unknown",
            "误报 / 漏报的上报路径与一键回滚方案尚未定义。",
            "临床风险管理（ISO 14971 思路）",
            "定义不良事件上报链路与灰度回滚开关",
        ),
    ]

    level = _level(flags)
    if level == "低" and any(r.category == "safety" for r in reqs):
        level = "中"  # 只要碰了安全类需求，就不可能只是"普通内部工具"
    red_lines = _red_lines(flags, mentions_research)
    passed = sum(1 for f in findings if f["status"] == "pass")
    failed = sum(1 for f in findings if f["status"] == "fail")

    return {
        "level": level,
        "flags": flags,
        "flag_names": [FLAG_NAMES.get(f, f) for f in flags],
        "findings": findings,
        "red_lines": red_lines,
        "counts": {"pass": passed, "fail": failed, "unknown": len(findings) - passed - failed},
        "gate": _gate(level),
        "blocking_questions": _blocking_questions(flags),
        "disclaimer": "以上为产品侧风险提示，不构成法律或注册结论；最终以医院法务 / 法规部门意见为准。",
    }


def _finding(cid: str, title: str, status: str, detail: str, basis: str, action: str) -> dict[str, str]:
    return {"id": cid, "title": title, "status": status, "detail": detail, "basis": basis, "action": action}


def _level(flags: list[str]) -> str:
    if "clinical_advice" in flags or "decision_support" in flags:
        return "高"
    if "phi" in flags or "patient_facing" in flags:
        return "中"
    return "低"


def _gate(level: str) -> str:
    if level == "高":
        return "暂不进入开发：先完成器械属性判定与人审设计评审，否则做出来也不能用。"
    if level == "中":
        return "可进入原型验证：但脱敏方案与边界说明必须先于真实数据接入。"
    return "可进入开发：保持人审留痕，按普通内部工具流程推进。"


def _red_lines(flags: list[str], research: bool) -> list[dict[str, str]]:
    ids = {"RL-04", "RL-06"}
    if "clinical_advice" in flags:
        ids.add("RL-01")
    if "phi" in flags or "export" in flags:
        ids.add("RL-02")
    if "decision_support" in flags:
        ids.add("RL-03")
    if research:
        ids.add("RL-05")
    return [r for r in T.RED_LINES if r["id"] in ids]


def _blocking_questions(flags: list[str]) -> list[str]:
    qs: list[str] = []
    if "clinical_advice" in flags:
        qs.append("这个功能是给医生「看证据、做选择」，还是直接给出治疗建议？前者是工具，后者是诊疗行为。")
    if "phi" in flags:
        qs.append("入模数据能不能做到只留必要字段？脱敏后信息是否还够用？")
    if "patient_facing" in flags:
        qs.append("患者看到的每一句话，谁签字负责？出了问题怎么追溯？")
    if "decision_support" in flags:
        qs.append("如果模型漏提示了一个高危患者，责任怎么算？这个漏报率临床上可接受吗？")
    if not qs:
        qs.append("这个能力出错的后果是什么？最坏情况下谁来兜底？")
    return qs
