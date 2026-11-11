"""评测：在人工标注的金标集上量化抽取质量。

刻意做了 dev / test 切分：
- dev（INT-001~003）是我写词表时看过的样本，指标必然好看，不能拿来吹；
- test（INT-004~006）是词表冻结后才写的，指标才代表泛化能力。

一个不做切分的评测等于自证。这个项目里最诚实的数字，都在 test 那几行。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import pipeline

ORD = {"Must": 3, "Should": 2, "Could": 1, "Won't": 0}


def _corpus(req) -> str:
    return " ".join([req.scene, req.statement] + [e["quote"] for e in req.evidence])


def evaluate_one(gold: dict) -> dict:
    iv = pipeline.load_interview(pipeline.INTERVIEW_DIR / f"{gold['interview_id']}.json")
    res = pipeline.analyze(iv)
    corpus = {r.id: _corpus(r) for r in res.requirements}

    covered = 0
    prio_hit = 0
    missed: list[str] = []
    for theme in gold["themes"]:
        matched = [r for r in res.requirements if any(m in corpus[r.id] for m in theme["match_any"])]
        if not matched:
            missed.append(theme["name"])
            continue
        covered += 1
        best = max(matched, key=lambda r: ORD[r.moscow])
        if ORD[best.moscow] >= ORD[theme["priority"]]:
            prio_hit += 1

    hit_reqs = [
        r
        for r in res.requirements
        if any(any(m in corpus[r.id] for m in t["match_any"]) for t in gold["themes"])
    ]

    expected = set(gold.get("expected_flags", []))
    detected = set(res.compliance["flags"])

    return {
        "id": gold["interview_id"],
        "split": gold.get("split", "dev"),
        "themes": len(gold["themes"]),
        "reqs": len(res.requirements),
        "covered": covered,
        "prio_hit": prio_hit,
        "req_hit": len(hit_reqs),
        "flag_total": len(expected),
        "flag_hit": len(expected & detected),
        "missed": missed,
        "stated_status": str(res.priority_check.get("status", "")),
    }


def _ratio(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0


def run_eval(verbose: bool = True) -> int:
    if not pipeline.GOLD_DIR.exists():
        print("找不到金标集 data/gold/")
        return 1

    rows = [evaluate_one(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(pipeline.GOLD_DIR.glob("*.json"))]
    splits: dict[str, list[dict]] = {}
    for r in rows:
        splits.setdefault(r["split"], []).append(r)

    if verbose:
        print("ID        split  themes  reqs  recall  prio   prec   flag")
        for r in rows:
            flag = f"{_ratio(r['flag_hit'], r['flag_total']):>6.2f}" if r["flag_total"] else "     -"
            print(
                f"{r['id']:<9} {r['split']:<6} {r['themes']:>6} {r['reqs']:>5} "
                f"{_ratio(r['covered'], r['themes']):>6.2f} {_ratio(r['prio_hit'], r['themes']):>6.2f} "
                f"{_ratio(r['req_hit'], r['reqs']):>6.2f} {flag}"
            )
        print("-" * 62)
        for name, group in splits.items():
            t = sum(g["themes"] for g in group)
            q = sum(g["reqs"] for g in group)
            flagged = [g for g in group if g["flag_total"]]
            flag_txt = (
                f"{_ratio(sum(g['flag_hit'] for g in flagged), sum(g['flag_total'] for g in flagged)):>6.2f}"
                if flagged
                else "     -"
            )
            print(
                f"{name.upper():<9} {'':<6} {t:>6} {q:>5} "
                f"{_ratio(sum(g['covered'] for g in group), t):>6.2f} "
                f"{_ratio(sum(g['prio_hit'] for g in group), t):>6.2f} "
                f"{_ratio(sum(g['req_hit'] for g in group), q):>6.2f} {flag_txt}"
            )

    report = _render_report(rows, splits)
    pipeline.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = pipeline.OUTPUT_DIR / "eval_report.md"
    path.write_text(report, encoding="utf-8")
    if verbose:
        print(f"\n报告已写出：{path}")
    return 0


def _render_report(rows: list[dict], splits: dict[str, list[dict]]) -> str:
    L: list[str] = ["# 评测报告", ""]
    L.append("> 由 `python run.py eval` 自动生成。金标集为人工标注的「理想需求集」，")
    L.append("> 指标含义见 docs/04-评测方案.md。")
    L.append("")
    L.append("## 1. 汇总")
    L.append("")
    L.append("| 数据集 | 访谈数 | 金标主题 | 生成需求 | 需求召回 | 优先级命中 | 需求精确率 | 合规召回 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, group in splits.items():
        t = sum(g["themes"] for g in group)
        q = sum(g["reqs"] for g in group)
        flagged = [g for g in group if g["flag_total"]]
        flag_cell = (
            f"{_ratio(sum(g['flag_hit'] for g in flagged), sum(g['flag_total'] for g in flagged)):.2f}"
            if flagged
            else "—"
        )
        L.append(
            f"| {name} | {len(group)} | {t} | {q} | "
            f"{_ratio(sum(g['covered'] for g in group), t):.2f} | "
            f"{_ratio(sum(g['prio_hit'] for g in group), t):.2f} | "
            f"{_ratio(sum(g['req_hit'] for g in group), q):.2f} | {flag_cell} |"
        )
    L.append("")
    L.append("## 2. 逐访谈明细")
    L.append("")
    L.append("| ID | 数据集 | 金标主题 | 生成需求 | 召回 | 优先级命中 | 精确率 | 一致性校验 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        L.append(
            f"| {r['id']} | {r['split']} | {r['themes']} | {r['reqs']} | "
            f"{_ratio(r['covered'], r['themes']):.2f} | {_ratio(r['prio_hit'], r['themes']):.2f} | "
            f"{_ratio(r['req_hit'], r['reqs']):.2f} | {r['stated_status']} |"
        )
    L.append("")
    L.append("## 3. 失败分析")
    L.append("")
    for r in rows:
        if r["missed"]:
            L.append(f"- **{r['id']}（{r['split']}）** 漏掉：{'、'.join(r['missed'])}")
    L.append("")
    L.append("### 已定位的系统性失败模式")
    L.append("")
    L.append("1. **隐喻与口语没被识别**：急诊「从一百个里把这几个人挑出来」= 分诊，但词表里没有「挑」，规则版直接漏掉。")
    L.append("2. **一句话里两个痛点只取一个**：「抢救记录事后补 + 时间点靠估」同时是效率和准确性，聚类强制二选一。")
    L.append("3. **流程协调类没有对应类别**：骨科「排期撞车」属于跨部门协调，既不是效率也不是信息，规则版几乎全漏。")
    L.append("4. **领域名词不在表里**：「危急值」「耗材」「手术排期」都是标准临床术语，但词表是按通用中文写的。")
    L.append("")
    L.append("这四条就是 v0.2 换 LLM 抽取的直接理由，也是评测集存在的意义：")
    L.append("**没有这组数字，换模型就是拍脑袋；有了这组数字，换完能证明好多少。**")
    L.append("")
    return "\n".join(L)
