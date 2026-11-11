"""命令行入口。

    python run.py list
    python run.py analyze INT-001
    python run.py demo            # 全量分析并写出 output/
    python run.py eval            # 跑金标评测
    python run.py web             # 起演示界面
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import llm, pipeline

CAT = {
    "efficiency": "效率",
    "toil": "重复劳动",
    "information": "信息",
    "safety": "安全",
    "communication": "沟通",
    "data": "数据",
}


def _polish_fn(enabled: bool):
    if not enabled:
        return None
    if not llm.available():
        print("[warn] 未检测到 CLINSIGHT_LLM_KEY / OPENAI_API_KEY，跳过润色，规则结果不受影响。")
        return None
    return llm.rewrite


def cmd_list(_args) -> int:
    for iv in pipeline.list_interviews():
        print(f"{iv.id}  {iv.title}  {len(iv.transcript)} 轮  {iv.duration_min} 分钟")
    return 0


def cmd_analyze(args) -> int:
    path = Path(args.id)
    if not path.exists():
        path = pipeline.INTERVIEW_DIR / f"{args.id}.json"
    if not path.exists():
        print(f"找不到访谈：{args.id}", file=sys.stderr)
        return 1

    interview = pipeline.load_interview(path)
    polish = _polish_fn(args.llm) if args.llm else None
    mode = args.mode
    if args.llm and polish:
        mode = "hybrid" if mode == "rule" else mode
    result = pipeline.analyze(interview, mode=mode, polish=polish, use_cache=not args.no_cache)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"\n=== {result.interview_title} · {result.interview_id} · {result.mode} ===")
    print(f"[抽取] {result.stats['utterances']} 轮对话 → {len(result.pain_points)} 个痛点簇 → {len(result.requirements)} 条需求\n")

    print("痛点簇：")
    for p in result.pain_points:
        print(f"  {p.id} [{p.scene}/{CAT.get(p.category, p.category)}] {p.phenomenon}（{p.evidence_count} 条证据）")

    print("\n需求（按 MoSCoW）：")
    for r in sorted(result.requirements, key=lambda x: -x.score):
        gate = " ⛔" if r.compliance_flags else ""
        print(f"  {r.id} {r.moscow:<7} {r.score:>5}  {r.statement[:56]}{gate}")

    c = result.compliance
    print(f"\n合规：{c['level']}风险 · 通过 {c['counts']['pass']} / 风险 {c['counts']['fail']} / 待确认 {c['counts']['unknown']}")
    print(f"  → {c['gate']}")

    pc = result.priority_check
    if pc:
        icon = {"aligned": "[一致]", "conflict": "[冲突]", "unknown": "[待确认]"}.get(str(pc.get("status")), "")
        print(f"\n一致性 {icon} {pc['note']}")
    print(f"\n耗时 {result.stats['elapsed_ms']} ms")

    if args.out:
        written = pipeline.write_outputs([result], args.out)
        for w in written:
            print(f"已写出：{w}")
    return 0


def cmd_demo(args) -> int:
    results = pipeline.analyze_all()
    written = pipeline.write_outputs(results)
    for r in results:
        must = [x.id for x in r.requirements if x.moscow == "Must"]
        print(f"{r.interview_id}  {r.interview_title}  需求 {len(r.requirements)}  Must {len(must)}  合规 {r.compliance['level']}")
    print(f"\n共写出 {len(written)} 个文件到 {pipeline.OUTPUT_DIR}")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import run_eval

    return run_eval(verbose=not args.quiet)


def cmd_ab(args) -> int:
    from .ab import run_ab

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
    return run_ab(modes=modes, use_cache=not args.no_cache)


def cmd_web(args) -> int:
    from .web import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clinsight", description="临床需求洞察 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出全部访谈").set_defaults(func=cmd_list)

    a = sub.add_parser("analyze", help="分析一份访谈")
    a.add_argument("id", help="访谈 ID 或 json 路径")
    a.add_argument("--json", action="store_true", help="输出完整 JSON")
    a.add_argument("--out", help="写出 PRD 与 JSON 的目录")
    a.add_argument("--mode", choices=["rule", "llm", "hybrid"], default="rule", help="抽取方式")
    a.add_argument("--no-cache", action="store_true", help="忽略 LLM 结果缓存，强制重新调用")
    a.add_argument("--llm", action="store_true", help="用大模型润色需求陈述（可选，失败自动降级）")
    a.set_defaults(func=cmd_analyze)

    ab = sub.add_parser("ab", help="跑规则 vs LLM 的 A/B 对照实验")
    ab.add_argument("--modes", help="逗号分隔：rule,llm,hybrid（默认全跑）")
    ab.add_argument("--no-cache", action="store_true")
    ab.set_defaults(func=cmd_ab)

    d = sub.add_parser("demo", help="全量分析并写出 output/")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="在金标集上评测")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_eval)

    w = sub.add_parser("web", help="启动本地演示界面")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(func=cmd_web)

    l = sub.add_parser("list-interviews", help=argparse.SUPPRESS)
    l.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
