"""命令行入口。

单张打分（只需要 AI 的图纸 + 给 AI 的 STEP；标准答案 --ref 可选）：
  python -m drawing_grader grade --cand ai.png --step case44.step [--out results/one]

批量打分（清单 CSV，列：case_id,generator,candidate,step,reference,model_images,inventory；
step / reference / model_images / inventory 都可留空；model_images 多张用 ; 分隔；路径相对于 CSV 所在目录）：
  python -m drawing_grader batch manifest.csv -o results/ --workers 2

打分器自检（oracle / null / mutation，确认打分器本身可靠）：
  python -m drawing_grader check --ref-image 合格图纸.png --step case44.step

把图纸抽取成 JSON（制作/校对参考答案用）：
  python -m drawing_grader extract drawing.png -o drawing.json

准确度与额度：
  --samples N         读图采样 N 次投票合并（默认 1；3 次更准，读图额度 ×3）
  --judge-samples N   3D 核对采样 N 次（默认 1）

调用 Claude 的方式（环境变量）：
  DRAWING_GRADER_BACKEND=cli   用本机 Claude Code（claude -p），走订阅额度（无 API key 时的默认）
  DRAWING_GRADER_BACKEND=api   用 Anthropic API（设置了 ANTHROPIC_API_KEY 时的默认）
  DRAWING_GRADER_EFFORT=high   思考强度 low/medium/high/xhigh/max
每次调用的耗时/用量记录在 .grader_cache/usage.jsonl

每次打分输出（Harbor 风格）：
  reward.json          {"reward": 总分 0~1, 各部分得分...}
  reward-details.json  每个评分项的 id / 分数 / 理由，门槛结果，采样一致率
  report.json          完整报告      extraction.json  读图结果
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .grader import COMPONENT_WEIGHTS, CaseReport, grade_files

NAME_CN = {
    "accuracy": "准确性(STEP)", "completeness": "完整性(STEP)", "reference": "与参考图比对",
    "fidelity": "与3D模型一致", "standard": "规格完整性",
    "consistency": "内部一致性", "layout": "排版质量",
    "views": "视图", "dimensions": "尺寸", "holes": "孔/螺纹标注", "gdt": "形位公差", "datums": "基准",
    "chamfers": "倒角", "roughness": "粗糙度", "title_block": "标题栏", "tech_requirements": "技术要求",
}


def print_report(rep: dict, verbose: bool) -> None:
    print(f"\n总分: {rep['total']:.1f} / 100")
    if rep["capped_by"]:
        print(f"  !! {rep['capped_by']}")
    if rep["agreement"]:
        flag = "  -> 读数不稳定，建议人工复核" if rep["needs_review"] else ""
        print("  采样一致率: " + ", ".join(f"{k} {v:.0%}" for k, v in rep["agreement"].items()) + flag)
    print()
    for c in rep["components"]:
        print(f"  {NAME_CN.get(c['name'], c['name']):<12}{c['score']:>6.1f}   (权重 {c['weight']:g})")
    if not verbose:
        return
    failed_gates = [g for g in rep["gates"] if not g["passed"]]
    if failed_gates:
        print("\n== 门槛 ==")
        for g in failed_gates:
            print(f"  - {g['id']}：{g['reason']}")
    for c in rep["components"]:
        for d in c["details"]:
            partial = [m for m in d["matched"] if "部分正确" in m]
            if not (d["missing"] or d["extra"] or partial):
                continue
            print(f"\n== {NAME_CN.get(c['name'], c['name'])} / {NAME_CN.get(d['name'], d['name'])}  {d['score']:.1f}% ==")
            for m in partial:
                print(f"  ~ {m}")
            for m in d["missing"]:
                print(f"  - {m}")
            for e in d["extra"]:
                print(f"  + {e}")
            for n in d["notes"]:
                print(f"  * {n}")


def write_outputs(out: Path, rep: CaseReport, extraction) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    dump("reward.json", rep.reward())
    dump("reward-details.json", rep.reward_details())
    dump("report.json", rep.to_dict())
    (out / "extraction.json").write_text(extraction.model_dump_json(indent=2), encoding="utf-8")


def load_weights(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def cmd_grade(args) -> None:
    rep, cand = grade_files(args.cand, args.model, args.ref, load_weights(args.weights), args.inventory,
                            args.samples, args.judge_samples, args.step)
    print_report(rep.to_dict(), verbose=not args.quiet)
    if args.out:
        write_outputs(Path(args.out), rep, cand)
        print(f"\n结果已写入 {args.out}/（reward.json, reward-details.json, report.json, extraction.json）")


def cmd_batch(args) -> None:
    manifest = Path(args.manifest)
    base = manifest.parent
    out = Path(args.out)
    weights = load_weights(args.weights)

    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def resolve(p: str | None) -> Path | None:
        p = (p or "").strip()
        return (base / p) if p else None

    def run(i: int, row: dict) -> dict:
        case, gen = row.get("case_id", "").strip(), row.get("generator", "").strip()
        rid = f"{i:04d}_{case}_{gen}".strip("_")
        record = {"row": i, "case_id": case, "generator": gen, "candidate": row.get("candidate", "")}
        try:
            models = [resolve(x) for x in (row.get("model_images") or "").split(";") if x.strip()]
            rep, cand = grade_files(resolve(row["candidate"]), models or None, resolve(row.get("reference")),
                                    weights, resolve(row.get("inventory")), args.samples, args.judge_samples,
                                    resolve(row.get("step")))
            write_outputs(out / "trials" / rid, rep, cand)
            d = rep.to_dict()
            record.update(total=d["total"], needs_review=d["needs_review"], capped_by=d["capped_by"], error="")
            for c in d["components"]:
                record[c["name"]] = c["score"]
        except Exception as e:  # 单行失败不影响其他行；与 agent 失败区分开，记为 error 而不是 0 分
            record.update(total="", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        return record

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run, i, r) for i, r in enumerate(rows)]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            status = f"{r['total']:.1f}" if r["error"] == "" else "失败"
            print(f"[{k}/{len(rows)}] {r['case_id']} {r['generator']} {r['candidate']}: {status}", file=sys.stderr)
            results.append(r)
    results.sort(key=lambda r: r["row"])

    out.mkdir(parents=True, exist_ok=True)
    cols = ["row", "case_id", "generator", "candidate", "total", *COMPONENT_WEIGHTS, "needs_review", "capped_by", "error"]
    with (out / "scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    by_gen: dict[str, list[float]] = {}
    for r in results:
        if r["error"] == "":
            by_gen.setdefault(r["generator"] or "-", []).append(r["total"])
    n_err = sum(r["error"] != "" for r in results)
    n_rev = sum(bool(r.get("needs_review")) for r in results)
    print(f"\n完成 {len(results)} 行，失败 {n_err} 行，需人工复核 {n_rev} 行。结果: {out / 'scores.csv'}")
    for g, xs in sorted(by_gen.items()):
        print(f"  {g:<20} 平均 {sum(xs) / len(xs):6.1f}  (n={len(xs)})")


def cmd_check(args) -> None:
    from .selfcheck import run_check

    results = run_check(args.ref_image, args.ref, args.model, args.inventory, args.samples, args.judge_samples,
                        load_weights(args.weights), args.step)
    print(f"\n{'检查项':<34}{'期望':>10}{'实际':>8}   结果")
    for r in results:
        print(f"{r.test:<34}{r.expect:>10}{r.actual:>8.1f}   {'PASS' if r.passed else 'FAIL'}  {r.note}")
    n_fail = sum(not r.passed for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} 通过")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    sys.exit(1 if n_fail else 0)


def cmd_extract(args) -> None:
    from .extract import extract_drawing

    d = extract_drawing(args.image, args.samples)
    Path(args.out).write_text(d.model_dump_json(indent=2), encoding="utf-8")
    print(f"已写入 {args.out}")


def main(argv: list[str] | None = None) -> None:
    # Windows 终端默认 GBK 编码，打印 Ø、↧ 等符号会报错，统一改成 UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="drawing_grader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_sampling(p):
        p.add_argument("--samples", type=int, default=1, help="读图采样次数（默认 1，3 更准）")
        p.add_argument("--judge-samples", type=int, default=1, help="3D 核对采样次数（默认 1）")

    g = sub.add_parser("grade", help="对一张待评图纸打分")
    g.add_argument("--cand", required=True, help="待评图纸图片（或已抽取的 JSON）")
    g.add_argument("--model", action="append", help="3D 模型图片，可重复多次传多角度")
    g.add_argument("--inventory", help="预先生成的 3D 特征清单 JSON（可选）")
    g.add_argument("--ref", help="参考图纸图片或 JSON（可选）")
    g.add_argument("--step", help="给 AI 的 STEP 文件（可选，提供时核对每个数值是否真实存在）")
    g.add_argument("--weights", help='组成部分权重 JSON，如 {"fidelity": 40}')
    g.add_argument("--out", help="输出目录（reward.json / reward-details.json / report.json / extraction.json）")
    g.add_argument("-q", "--quiet", action="store_true")
    add_sampling(g)
    g.set_defaults(fn=cmd_grade)

    b = sub.add_parser("batch", help="按清单 CSV 批量打分")
    b.add_argument("manifest")
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--weights")
    add_sampling(b)
    b.set_defaults(fn=cmd_batch)

    c = sub.add_parser("check", help="打分器自检：oracle / null / mutation")
    c.add_argument("--ref-image", required=True, help="一张已知合格的图纸（用作 oracle）")
    c.add_argument("--ref", help="标准答案 JSON 或图片（可选；不给则只按 STEP 打分）")
    c.add_argument("--model", action="append", help="3D 模型图片")
    c.add_argument("--inventory", help="3D 特征清单 JSON（可选）")
    c.add_argument("--step", help="STEP 文件（可选）")
    c.add_argument("--weights")
    c.add_argument("--out", help="自检结果 JSON")
    add_sampling(c)
    c.set_defaults(fn=cmd_check)

    e = sub.add_parser("extract", help="把图纸图片抽取为 JSON")
    e.add_argument("image")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--samples", type=int, default=1)
    e.set_defaults(fn=cmd_extract)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
