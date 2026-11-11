"""v0.2 的 A/B 实验：规则 vs 纯 LLM 抽取 vs 并集。

设计要点（比代码重要）：

1. **单变量**：只换抽取器，可行性/合规/排序/金标口径全部不变。
   否则"涨分了"说不清是哪一环贡献的。
2. **同一测试集**：dev/test 切分沿用 v0.1，test 仍未参与任何调参。
3. **降级的样本不进分母**：某份访谈走了 fallback（没 key / 调用失败 / 输出不合格），
   它跑出来的其实是规则版结果。把它算进 LLM 的平均分，会得到一张"看起来一样"的假表。
4. **没跑出真实结果就明确说"未运行"**，不生成数字。宁可交白卷，不交假卷。
"""

from __future__ import annotations

import json

from . import evaluate, pipeline

MODES: list[tuple[str, str]] = [
    ("rule", "A 规则版（v0.1 基线）"),
    ("llm", "B 纯 LLM 抽取"),
    ("hybrid", "C LLM + 规则并集"),
]


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else round(a / b, 3)


def aggregate(rows: list[dict], split: str) -> dict | None:
    group = [r for r in rows if r["split"] == split]
    if not group:
        return None
    themes = sum(g["themes"] for g in group)
    reqs = sum(g["reqs"] for g in group)
    flagged = [g for g in group if g["flag_total"]]
    return {
        "n": len(group),
        "reqs": reqs,
        "recall": _ratio(sum(g["covered"] for g in group), themes),
        "precision": _ratio(sum(g["req_hit"] for g in group), reqs),
        "prio": _ratio(sum(g["prio_hit"] for g in group), themes),
        "flags": _ratio(sum(g["flag_hit"] for g in flagged), sum(g["flag_total"] for g in flagged))
        if flagged
        else None,
    }


def run_ab(modes: list[str] | None = None, use_cache: bool = True, verbose: bool = True) -> int:
    chosen = [m for m in MODES if (modes is None or m[0] in modes)]
    golds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pipeline.GOLD_DIR.glob("*.json"))]
    if not golds:
        print("找不到金标集 data/gold/")
        return 1

    per_mode: dict[str, dict] = {}
    for key, label in chosen:
        ok: list[dict] = []
        fell_back: list[str] = []
        accepted = rejected = 0
        for gold in golds:
            iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / f"{gold['interview_id']}.json")
            res = pipeline.analyze(iv, mode=key, use_cache=use_cache)
            info = res.stats.get("llm") if isinstance(res.stats.get("llm"), dict) else {}
            bad = bool(info.get("fallback") or info.get("llm_error")) if info else False
            accepted += int(info.get("accepted", 0) or 0)
            rejected += int(info.get("rejected", 0) or 0)
            row = evaluate.score(
                gold, res.requirements, set(res.compliance["flags"]), str(res.priority_check.get("status", ""))
            )
            row["id"] = gold["interview_id"]
            row["split"] = gold.get("split", "dev")
            (fell_back.append(gold["interview_id"]) if bad else ok.append(row))
        per_mode[key] = {
            "label": label,
            "ok": ok,
            "fell_back": fell_back,
            "accepted": accepted,
            "rejected": rejected,
            "total": len(golds),
        }

    if verbose:
        _print_table(chosen, per_mode, golds)

    path = pipeline.OUTPUT_DIR / "ab_report.md"
    pipeline.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(chosen, per_mode), encoding="utf-8")

    if verbose:
        print(f"\n报告已写出：{path}")

    llm_modes = [k for k, _ in chosen if k != "rule"]
    not_run = [k for k in llm_modes if not per_mode[k]["ok"]]
    if not_run and verbose:
        print("\n[!] 以下模式未产生任何有效样本，未计入对比：" + "、".join(not_run))
        print("    原因通常是缺少 CLINSIGHT_LLM_KEY，或模型调用全部失败。")
        print("    这里不会用降级结果冒充 LLM 结果。")
    if llm_modes and len(not_run) == len(llm_modes):
        return 2
    return 0


def _cell(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _print_table(chosen, per_mode, golds) -> None:
    split_total: dict[str, int] = {}
    for g in golds:
        split_total[g.get("split", "dev")] = split_total.get(g.get("split", "dev"), 0) + 1

    print(f"{'抽取方式':<20}{'数据集':<8}{'需求':>5}{'召回':>7}{'精确':>7}{'优先级':>8}{'有效':>8}")
    print("-" * 64)
    for key, label in chosen:
        m = per_mode[key]
        if key != "rule" and not m["ok"]:
            print(f"{label:<20}{'—':<8}{'未运行':>10}  （{len(m['fell_back'])}/{m['total']} 次降级）")
            continue
        for split in ("dev", "test"):
            agg = aggregate(m["ok"], split)
            if not agg:
                continue
            eff = f"{agg['n']}/{split_total.get(split, agg['n'])}"
            print(
                f"{label:<20}{split:<8}{agg['reqs']:>5}{_cell(agg['recall']):>7}"
                f"{_cell(agg['precision']):>7}{_cell(agg['prio']):>8}{eff:>8}"
            )
    print("-" * 64)
    for key, label in chosen:
        m = per_mode[key]
        if key == "rule":
            continue
        print(f"{label}: schema 通过 {m['accepted']} 条 / 拒绝 {m['rejected']} 条 / 降级 {len(m['fell_back'])} 条")


def _render(chosen, per_mode) -> str:
    L = ["# A/B 实验报告（v0.2）", ""]
    L.append("> 由 `python run.py ab` 生成。单变量对照：只换抽取器，其余环节与 v0.1 完全一致。")
    L.append("> 降级的样本不计入分母——否则跑出来的是规则版结果，却挂着 LLM 的名字。")
    L.append("")

    pending = [k for k, _ in chosen if k != "rule" and not per_mode[k]["ok"]]
    if pending:
        L.append("> ⚠️ **以下模式未运行，本报告不含其数字**：" + "、".join(per_mode[k]["label"] for k in pending))
        L.append("> 复现方法见 README「v0.2 实验复现」。")
        L.append("")

    L.append("## 1. 汇总")
    L.append("")
    L.append("| 抽取方式 | 数据集 | 生成需求 | 需求召回 | 需求精确率 | 优先级命中 | 有效样本 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in chosen:
        m = per_mode[key]
        if key != "rule" and not m["ok"]:
            L.append(f"| {label} | — | 未运行 | — | — | — | 0/{m['total']} |")
            continue
        for split in ("dev", "test"):
            agg = aggregate(m["ok"], split)
            if not agg:
                continue
            L.append(
                f"| {label} | {split} | {agg['reqs']} | {_cell(agg['recall'])} | "
                f"{_cell(agg['precision'])} | {_cell(agg['prio'])} | {agg['n']} |"
            )
    L.append("")
    L.append("## 2. 逐访谈明细（仅统计有效样本）")
    L.append("")
    for key, label in chosen:
        m = per_mode[key]
        L.append(f"### {label}")
        L.append("")
        if not m["ok"]:
            L.append("未运行。降级样本：" + ("、".join(m["fell_back"]) or "无"))
            L.append("")
            continue
        L.append("| ID | 数据集 | 金标主题 | 生成需求 | 召回 | 精确率 | 一致性 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in m["ok"]:
            L.append(
                f"| {r['id']} | {r['split']} | {r['themes']} | {r['reqs']} | "
                f"{_ratio(r['covered'], r['themes']) or 0:.2f} | "
                f"{_ratio(r['req_hit'], r['reqs']) or 0:.2f} | {r['stated_status']} |"
            )
        if m["fell_back"]:
            L.append("")
            L.append("降级（未计入）：" + "、".join(m["fell_back"]))
        L.append("")
    L.append("## 3. 输出校验")
    L.append("")
    L.append("| 模式 | schema 通过 | 被拒 | 降级 |")
    L.append("| --- | --- | --- | --- |")
    for key, label in chosen:
        m = per_mode[key]
        if key == "rule":
            L.append(f"| {label} | — | — | 0 |")
        else:
            L.append(f"| {label} | {m['accepted']} | {m['rejected']} | {len(m['fell_back'])}/{m['total']} |")
    L.append("")
    L.append("> 被拒 = 证据对不上原文 / 类别非法 / 陈述过短。**宁可丢，不可编。**")
    L.append("")
    return "\n".join(L)
