"""【有参考图纸时】把待评图纸与参考图纸逐项比对，按类别打分（0-100）。纯规则，结果确定可复现。

每个列表类别：先按相似度做最优二分匹配（Hungarian），再算软 F1：
    P = Σsim / |cand|，R = Σsim / |ref|，F1 = 2PR/(P+R)
漏标会拉低 R，多标/错标会拉低 P。

公平原则：只比对能从 STEP 几何得出的信息。
  比对：视图类型/剖面线、尺寸数值、孔（直径、数量、深度、通孔、双面）、倒角
  不比对（STEP 中没有，属于设计者决定）：尺寸公差与配合、螺纹规格与螺纹深度、形位公差与基准、
          粗糙度、技术要求、标题栏内容，以及视图/剖切字母的命名（A-A 还是 E-E 是任意的）
写法归一化：
- 直径尺寸（Ø20）一律按孔标注比对，所以 "Ø20" 能与参考的 "2×Ø20 THRU" 匹配（缺数量/THRU 只给部分分）
- 双面孔：参考写 "16× … BOTH SIDES"、图纸只标一面 "8×" 并注明背面相同，视为数量一致
提供 STEP 时：参考图没有、但 STEP 证实真实存在的尺寸/孔不算"多标"（参考图可能漏标，标注起点也可以不同）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import Chamfer, Dimension, Drawing, HoleCallout, View

DEFAULT_WEIGHTS: dict[str, float] = {
    "views": 20,
    "dimensions": 35,
    "holes": 35,
    "chamfers": 10,
}

EPS = 1e-3
LAYOUT_PENALTY_PER_ISSUE = 0.10


@dataclass
class CategoryResult:
    name: str
    score: float  # 0..1
    weight: float
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 逐条评分项（RewardKit 风格）：{"id", "score"(0..1), "reason"}，汇总到 reward-details.json
    items: list[dict] = field(default_factory=list)

    def add_item(self, item_id: str, score: float, reason: str = "") -> None:
        self.items.append({"id": f"{self.name}:{item_id}", "score": round(float(score), 3), "reason": reason})


@dataclass
class Report:
    total: float  # 0..100
    categories: list[CategoryResult]


# ---------- 基础比较 ----------

def _eq(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= EPS


# ---------- 各类相似度（0..1，0 表示不能匹配）----------

def sim_view(r: View, c: View) -> float:
    if r.kind != c.kind:
        return 0.0
    return 0.8 + 0.2 * (r.hatched == c.hatched)


# 深度尺寸（4-↧35）和在剖视图里标的线性尺寸（35）是同一个几何量的两种写法
_KIND_GROUP = {"depth": "linear"}


def sim_dim(r: Dimension, c: Dimension) -> float:
    same_kind = _KIND_GROUP.get(r.kind, r.kind) == _KIND_GROUP.get(c.kind, c.kind)
    if not same_kind or not _eq(r.value, c.value):
        return 0.0
    return 0.9 + 0.1 * (r.count == c.count)


# 图纸注明"背面孔与正面相同"之类的说明
MIRROR_NOTE = re.compile(r"BOTH\s*SIDES|BACK|REAR|OPPOSITE|IDENTICAL|SYMMETRIC|背面|反面|两面|对称", re.I)


def _count_ok(r: HoleCallout, c: HoleCallout) -> bool:
    # 只有明确的双面孔，"标一面×2" 才与 "BOTH SIDES 总数" 等价
    if r.count == c.count:
        return True
    return (r.both_sides or c.both_sides) and (r.count == 2 * c.count or c.count == 2 * r.count)


def sim_hole(r: HoleCallout, c: HoleCallout, mirror: bool = False) -> float:
    if not _eq(r.diameter, c.diameter):
        return 0.0
    depth_ok = r.through == c.through and _eq(r.depth, c.depth)
    return (
        0.3
        + 0.3 * _count_ok(r, c)
        + 0.3 * depth_ok
        + 0.1 * (r.both_sides == c.both_sides or (mirror and r.both_sides))
    )


def sim_chamfer(r: Chamfer, c: Chamfer) -> float:
    if not (_eq(r.size, c.size) and _eq(r.angle, c.angle)):
        return 0.0
    return 0.8 + 0.2 * (r.count == c.count)


def _dims_to_holes(d: Drawing) -> tuple[list[Dimension], list[HoleCallout]]:
    """把直径尺寸并入孔标注，避免同一个孔因写法不同被双重扣分。"""
    dims = [x for x in d.dimensions if x.kind != "diameter"]
    holes = list(d.holes) + [
        HoleCallout(view=x.view, count=x.count, diameter=x.value, depth=None, through=False, thread=None,
                    thread_depth=None, both_sides=False, fit=None, tol_upper=None, tol_lower=None)
        for x in d.dimensions if x.kind == "diameter"
    ]
    return dims, holes


# ---------- 匹配与打分 ----------

def match_category(name, weight, ref: list, cand: list, sim: Callable, fmt: Callable = str,
                   extra_ok: Callable | None = None) -> CategoryResult:
    """extra_ok(c)：对参考图中没有对应项的图纸条目，若返回 True（如 STEP 证实真实存在）则不计入"多标"。"""
    res = CategoryResult(name=name, score=0.0, weight=weight)
    if not ref and not cand:
        res.score = 1.0
        return res
    if not ref or not cand:
        res.missing = [fmt(x) for x in ref]
        res.extra = [fmt(x) for x in cand]
        for x in res.missing:
            res.add_item(x, 0, "图上缺失")
        for x in res.extra:
            res.add_item(f"extra:{x}", 0, "参考图中没有")
        return res

    m = np.array([[sim(r, c) for c in cand] for r in ref])
    rows, cols = linear_sum_assignment(-m)
    used_r, used_c, total = set(), set(), 0.0
    for i, j in zip(rows, cols):
        if m[i, j] <= 0:
            continue
        used_r.add(i)
        used_c.add(j)
        total += m[i, j]
        tag = "" if m[i, j] >= 0.999 else f"  (部分正确 {m[i, j]:.2f}，图上为 {fmt(cand[j])})"
        res.matched.append(fmt(ref[i]) + tag)
        res.add_item(fmt(ref[i]), m[i, j], "正确" if not tag else tag.strip())
    res.missing = [fmt(r) for i, r in enumerate(ref) if i not in used_r]
    unmatched = [c for j, c in enumerate(cand) if j not in used_c]
    verified = [c for c in unmatched if extra_ok and extra_ok(c)]
    res.extra = [fmt(c) for c in unmatched if not (extra_ok and extra_ok(c))]
    for x in res.missing:
        res.add_item(x, 0, "图上缺失或数值不符")
    for x in res.extra:
        res.add_item(f"extra:{x}", 0, "参考图中没有对应项（多标或标错）")
    for c in verified:
        res.add_item(f"extra_verified:{fmt(c)}", 1, "参考图未标，但 STEP 证实真实存在，不扣分")
    if verified:
        res.notes.append(f"{len(verified)} 项参考图未标、但 STEP 证实存在的标注不计为多标："
                         + "、".join(fmt(c) for c in verified))

    n_cand = len(cand) - len(verified)
    p, r = (total / n_cand if n_cand else 1.0), total / len(ref)
    res.score = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return res


def _fmt_dim(d: Dimension) -> str:
    sym = {"diameter": "Ø", "radius": "R", "angle": "∠", "depth": "↧"}.get(d.kind, "")
    cnt = f"{d.count}-" if d.count > 1 else ""
    return f"[{d.view}] {cnt}{sym}{d.value:g}"


def _fmt_hole(h: HoleCallout) -> str:
    s = f"[{h.view}] {h.count}×Ø{h.diameter:g}"
    s += " THRU" if h.through else (f" ↧{h.depth:g}" if h.depth is not None else "")
    if h.both_sides:
        s += " BOTH SIDES"
    return s


def score_layout(cand: Drawing, weight: float) -> CategoryResult:
    n = len(cand.layout_issues)
    res = CategoryResult(
        name="layout",
        score=max(0.0, 1 - LAYOUT_PENALTY_PER_ISSUE * n),
        weight=weight,
        extra=list(cand.layout_issues),
        notes=[f"每个排版缺陷扣 {LAYOUT_PENALTY_PER_ISSUE:.0%}"],
    )
    for i, issue in enumerate(cand.layout_issues, 1):
        res.add_item(f"issue{i}", 0, issue)
    if not cand.layout_issues:
        res.add_item("no_issues", 1, "未发现排版缺陷")
    return res


def grade(ref: Drawing, cand: Drawing, weights: dict[str, float] | None = None, geo=None) -> Report:
    """geo：可选的 StepGeometry；提供时，参考图未标但 STEP 证实存在的尺寸/孔不算多标。"""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    dim_ok = hole_ok = None
    if geo is not None:
        from .step_geometry import verify_dimension, verify_hole

        dim_ok = lambda d: verify_dimension(geo, d.kind, d.value)
        hole_ok = lambda h: verify_hole(geo, h)[0] >= 0.999
    ref_dims, ref_holes = _dims_to_holes(ref)
    cand_dims, cand_holes = _dims_to_holes(cand)
    mirror = any(MIRROR_NOTE.search(t) for t in cand.tech_requirements) or any(h.both_sides for h in cand.holes)
    cats = [
        match_category("views", w["views"], ref.views, cand.views, sim_view,
                       lambda v: f"{v.label} ({v.kind}{', 剖面线' if v.hatched else ''})"),
        match_category("dimensions", w["dimensions"], ref_dims, cand_dims, sim_dim, _fmt_dim, dim_ok),
        match_category("holes", w["holes"], ref_holes, cand_holes, partial(sim_hole, mirror=mirror), _fmt_hole,
                       hole_ok),
        match_category("chamfers", w["chamfers"], ref.chamfers, cand.chamfers, sim_chamfer,
                       lambda c: f"[{c.view}] {c.count}×{c.size:g}×{c.angle:g}°"),
    ]
    total_w = sum(c.weight for c in cats)
    total = 100 * sum(c.score * c.weight for c in cats) / total_w
    return Report(total=total, categories=cats)
