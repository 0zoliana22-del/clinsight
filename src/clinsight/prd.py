"""交付层：把结构化结论渲染成一份可以被评审的 PRD。

刻意保留"取舍记录"和"开放问题"两节——面试里最容易被追问的不是你做了什么，
而是你为什么没做另一个。把没选的方案和理由写下来，比堆功能更能证明判断力。
"""

from __future__ import annotations

from .models import Interview, PainPoint, Requirement

MOSCOW_ORDER = ["Must", "Should", "Could", "Won't"]


def render(
    interview: Interview,
    points: list[PainPoint],
    reqs: list[Requirement],
    feasibility: dict[str, object],
    compliance: dict[str, object],
    priority_check: dict[str, object] | None = None,
) -> str:
    counts = {k: len([r for r in reqs if r.moscow == k]) for k in MOSCOW_ORDER}
    wont = counts["Won't"]
    musts = sorted([r for r in reqs if r.moscow == "Must"], key=lambda r: -r.score)
    scope = musts[0] if musts else (max(reqs, key=lambda r: r.score) if reqs else None)
    L: list[str] = []

    L.append(f"# {interview.department}「{scope.scene if scope else interview.setting}」AI 助手 · PRD v0.1")
    L.append("")
    evidence_total = sum(p.evidence_count for p in points)
    L.append(
        f"> 由 **ClinSight** 基于访谈 `{interview.id}`（{interview.role}，{interview.date}，"
        f"{interview.duration_min} 分钟）自动生成。共 {len(points)} 个痛点簇、{evidence_total} 条原始证据、"
        f"{len(reqs)} 条需求，全部可回溯到原话。"
    )
    L.append("")

    # ---------------------------------------------------------------- 0
    L.append("## 0. 一页结论")
    L.append("")
    if scope:
        L.append(f"- **第一版切口**：{scope.statement}")
    L.append(
        f"- **优先级分布**：Must {counts['Must']} / Should {counts['Should']} / "
        f"Could {counts['Could']} / Won't {wont}"
    )
    L.append(f"- **合规等级**：{compliance['level']}风险 —— {compliance['gate']}")
    L.append(f"- **技术判断**：{feasibility['summary']}")
    if priority_check:
        icon = {"aligned": "✅", "conflict": "⚠️", "unknown": "❓"}.get(str(priority_check.get("status")), "❓")
        L.append(f"- **一致性校验**（用户自述优先级 vs 模型排序）：{icon} {priority_check['note']}")
    L.append("")

    # ---------------------------------------------------------------- 1
    L.append("## 1. 背景与证据")
    L.append("")
    L.append("### 1.1 访谈信息")
    L.append("")
    L.append("| 项 | 内容 |")
    L.append("| --- | --- |")
    L.append(f"| 访谈对象 | {interview.role} |")
    L.append(f"| 科室 | {interview.department} |")
    L.append(f"| 主要场景 | {interview.setting} |")
    L.append(f"| 时长 | {interview.duration_min} 分钟 |")
    L.append(f"| 日期 | {interview.date} |")
    if interview.notes:
        L.append(f"| 备注 | {interview.notes} |")
    L.append("")
    L.append("### 1.2 痛点清单")
    L.append("")
    L.append("| ID | 场景 | 类别 | 现象 | 频次 | 严重度 | 证据 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for p in points:
        L.append(
            f"| {p.id} | {p.scene} | {_cat(p.category)} | {p.phenomenon} | {p.frequency} | "
            f"{p.severity}/5 | {p.evidence_count} 条 |"
        )
    L.append("")

    # ---------------------------------------------------------------- 2
    L.append("## 2. 目标与非目标")
    L.append("")
    L.append("### 2.1 目标（可验证）")
    L.append("")
    goals: list[str] = []
    for r in [r for r in reqs if r.moscow in ("Must", "Should")]:
        if r.acceptance[0] not in goals:
            goals.append(r.acceptance[0])
    for g in goals:
        L.append(f"- {g}")
    L.append("")
    L.append("### 2.2 非目标（本期明确不做）")
    L.append("")
    L.append("- 不做诊断结论、不给处方与剂量建议——这是红线，不是能力问题。")
    L.append("- 不替代医生签字确认，所有输出必须经人确认才进入病历。")
    L.append("- 不做全自动写入，先只做「生成草稿 + 人工确认」。")
    for r in [r for r in reqs if r.moscow == "Won't"]:
        L.append(f"- 本期不做：{r.statement}")
    L.append("")

    # ---------------------------------------------------------------- 3
    L.append("## 3. 需求清单（MoSCoW）")
    L.append("")
    if priority_check and priority_check.get("stated"):
        stated = priority_check["stated"]
        L.append(f"> 用户被问到「如果只能先做一个」时回答（原文第 {stated['turn']} 轮）：")
        L.append(f"> 「{stated['text']}」")
        L.append(">")
        L.append("> 这一段是后面所有排序的校验基准：模型可以有自己的判断，但不能假装没听见用户的选择。")
        L.append("")
    for level in MOSCOW_ORDER:
        group = [r for r in reqs if r.moscow == level]
        if not group:
            continue
        L.append(f"### {level}（{_level_cn(level)}，{len(group)} 条）")
        L.append("")
        for r in group:
            L.append(f"#### {r.id} · {r.scene} / {_cat(r.category)} · 得分 {r.score}")
            L.append("")
            L.append(f"**需求陈述**：{r.statement}")
            L.append("")
            L.append("**证据**：")
            for e in r.evidence[:3]:
                L.append(f"> [{e['turn']}] {e['speaker']}：「{e['quote']}」")
                L.append(">")
            L.append("**依据（可复算）**：")
            for line in r.rationale:
                L.append(f"- {line}")
            L.append("")
            L.append("**验收标准**：")
            for a in r.acceptance:
                L.append(f"- [ ] {a}")
            L.append("")
            spec = next((row for row in feasibility["rows"] if row["requirement_id"] == r.id), None)
            if spec:
                L.append(f"**AI 方案**：{spec['pattern']}")
                L.append(f"- 能做：{'；'.join(spec['can'])}")
                L.append(f"- 不能做：{'；'.join(spec['cannot'])}")
                L.append(f"- 人审点：{spec['human_check']}")
                L.append(f"- 评测指标：{'、'.join(spec['eval_metrics'])}")
                L.append("")

    # ---------------------------------------------------------------- 4
    L.append("## 4. 方案取舍记录")
    L.append("")
    L.append("| 决策点 | 备选 | 结论 | 理由 |")
    L.append("| --- | --- | --- | --- |")
    L.append(
        "| 第一版做检索还是全自动生成 | A. 端到端生成文书 B. 检索 + 带出处引用 | 选 B | "
        "生成类错误无法被用户快速识别，检索类错误点开原文就能发现；在信任建立前，可验证性 > 自动化程度 |"
    )
    L.append(
        "| 模型直出还是规则兜底 | A. 纯模型 B. 规则 + 模型双通道 | 选 B | "
        "漏报的代价远高于误报，纯模型无法承诺覆盖率；规则可解释、可审计，能过合规 |"
    )
    L.append(
        "| 先做小试点还是大版本 | A. 3 个月大版本 B. 2 周 5 人试点 | 选 B | "
        "需求真实性本身是最大不确定性，先用最小成本证伪，再谈投入 |"
    )
    L.append("")

    # ---------------------------------------------------------------- 5
    L.append("## 5. 合规评估")
    L.append("")
    L.append(f"**综合风险等级：{compliance['level']}** —— {compliance['gate']}")
    L.append("")
    if compliance["flags"]:
        L.append(f"触发信号：{'、'.join(compliance['flag_names'])}")
        L.append("")
    L.append("| 检查项 | 状态 | 说明 | 需要的动作 |")
    L.append("| --- | --- | --- | --- |")
    for f in compliance["findings"]:
        L.append(f"| {f['title']} | {_status(f['status'])} | {f['detail']} | {f['action']} |")
    L.append("")
    L.append("### 红线（不可越界）")
    L.append("")
    for rl in compliance["red_lines"]:
        L.append(f"- **{rl['id']} {rl['title']}**：{rl['detail']}（依据：{rl['basis']}）")
    L.append("")
    L.append(f"> {compliance['disclaimer']}")
    L.append("")

    # ---------------------------------------------------------------- 6
    L.append("## 6. 里程碑与协同")
    L.append("")
    L.append("| 周 | 目标 | 交付物 | 主责 / 协同 | 通过标准 |")
    L.append("| --- | --- | --- | --- | --- |")
    L.append(
        "| W1 | 需求基线锁定 | 需求清单签字版、验收口径 | PM / 科室主任、护士长 | 医生认领 Must 项，指标口径无异议 |"
    )
    L.append(
        "| W2 | 技术 PoC | 抽取 / 检索原型、20 条真实样本评测 | 算法 / 研发 | 字段准确率 ≥85%，引用命中率 ≥90% |"
    )
    L.append(
        "| W3 | 可用原型 | 可点交互、边界说明文案 | 研发 / 设计 / 法务 | 5 名目标用户走通主流程，SUS ≥70 |"
    )
    L.append(
        "| W4 | 小范围试点 | 2 周使用数据、评测报告 | 测试 / 合规 / PM | 核心指标较基线改善 ≥30%，合规清单 8/8 关闭 |"
    )
    L.append("")

    # ---------------------------------------------------------------- 7
    L.append("## 7. 风险与开放问题")
    L.append("")
    L.append("| 风险 | 影响 | 应对 |")
    L.append("| --- | --- | --- |")
    L.append("| 数据拿不到 / 接口不通 | PoC 无法按期 | W1 同步做接口盘点，拿不到就改「医生手动粘贴」的降级方案 |")
    L.append("| 模型幻觉被医生发现一次 | 信任崩塌，项目死亡 | 全量带出处 + 拒答机制 + 首屏写清覆盖范围 |")
    L.append("| 合规判定周期长 | 排期被卡 | 第一天就拉法规同事进群，不等开发完才送审 |")
    L.append("| 医生试用后不用 | 伪需求被证伪 | 这也是有效结论：2 周内记录弃用原因，及时转向 |")
    L.append("")
    L.append("**开放问题（需业务方回答）**：")
    L.append("")
    for q in compliance["blocking_questions"]:
        L.append(f"- {q}")
    L.append("")

    # ---------------------------------------------------------------- 8
    L.append("## 附录 A：证据原文（可回放）")
    L.append("")
    for p in points:
        L.append(f"**[{p.id}] {p.scene} / {_cat(p.category)}**")
        for e in p.evidence:
            L.append(f"- [{e['turn']}] {e['speaker']}：{e['quote']}")
        L.append("")

    return "\n".join(L)


def _cat(c: str) -> str:
    from .taxonomy import CATEGORY_LABEL

    return CATEGORY_LABEL.get(c, c)


def _level_cn(level: str) -> str:
    return {"Must": "必须有", "Should": "应该有", "Could": "可以有", "Won't": "本期不做"}.get(level, level)


def _status(s: str) -> str:
    return {"pass": "✅ 已覆盖", "fail": "❌ 有风险", "unknown": "⚠️ 待确认"}.get(s, s)
