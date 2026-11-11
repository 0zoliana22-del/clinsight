"""编排层：一次分析 = 7 个有序步骤，每步都可单独替换 / 单独评测。

    ingest → extract(pain) → build(req) → feasibility → compliance → prioritize → render

把"Agent"做成显式工作流而不是一个黑盒 prompt，理由是：
每一步的失败模式不同（抽漏了 / 判错了 / 排偏了），必须能定位到具体一步。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import compliance, extract, feasibility, prd, prioritize
from .models import Analysis, Interview

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INTERVIEW_DIR = DATA_DIR / "interviews"
GOLD_DIR = DATA_DIR / "gold"
OUTPUT_DIR = ROOT / "output"


def load_interview(path: str | Path) -> Interview:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Interview.from_dict(raw)


def list_interviews(directory: str | Path | None = None) -> list[Interview]:
    directory = Path(directory or INTERVIEW_DIR)
    items = [load_interview(p) for p in sorted(directory.glob("*.json"))]
    return items


def analyze(interview: Interview, mode: str = "rule", polish=None) -> Analysis:
    t0 = time.perf_counter()

    points = extract.extract_pain_points(interview)
    reqs = extract.build_requirements(points)

    feas = feasibility.assess(reqs)
    for r in reqs:
        compliance.assess_requirement(r, interview)
    comp = compliance.assess_interview(interview, reqs)
    prioritize.apply(reqs, points)
    check = prioritize.check_alignment(reqs, prioritize.stated_priority(interview))

    if polish:
        for r in reqs:
            better = polish(r.statement)
            if better:
                r.statement = better

    markdown = prd.render(interview, points, reqs, feas, comp, check)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    covered = sum(1 for r in reqs if len(r.evidence) >= 2)
    stats = {
        "utterances": len(interview.transcript),
        "pain_points": len(points),
        "requirements": len(reqs),
        "must": sum(1 for r in reqs if r.moscow == "Must"),
        "multi_evidence_ratio": round(covered / len(reqs), 2) if reqs else 0.0,
        "elapsed_ms": elapsed,
        "mode": mode,
    }
    return Analysis(
        interview_id=interview.id,
        interview_title=interview.title,
        mode=mode,
        pain_points=points,
        requirements=reqs,
        feasibility=feas,
        compliance=comp,
        priority_check=check,
        prd_markdown=markdown,
        stats=stats,
    )


def analyze_all(mode: str = "rule") -> list[Analysis]:
    return [analyze(iv, mode=mode) for iv in list_interviews()]


def write_outputs(results: list[Analysis], output_dir: str | Path | None = None) -> list[Path]:
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in results:
        md = out / f"PRD-{r.interview_id}.md"
        md.write_text(r.prd_markdown, encoding="utf-8")
        js = out / f"analysis-{r.interview_id}.json"
        js.write_text(json.dumps(r.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        written.extend([md, js])
    return written
