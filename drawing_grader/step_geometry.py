"""从 STEP 文件量出零件的全部真实几何，并自动推出"这张图纸必须标出什么"。

两个用途：
1. 准确性：图纸上每个数值是否在 STEP 中真实存在（标错扣分）
2. 完整性：按统一规则从 STEP 推出必须标注的项目（外形、各组孔、外圆、圆角、倒角、剖视），逐项检查是否标出
   —— 不需要每个零件单独准备标准答案

提取内容：
- 线性尺寸候选：包围盒、平行平面间距、直线边长度及其在各主方向上的投影、孔轴之间/孔轴到平面的距离、孔深
- 直径/半径：所有圆柱面
- 角度：平面之间的夹角
- 孔：按同轴圆柱面合并成"孔"，得到 直径、深度（含入口倒角）、是否通孔、数量
- 倒角：45° 锥面（入口倒角）的轴向尺寸

图纸上的数值只要与这些真实值之一相差不超过 VALUE_TOL，就认为是"真实存在的尺寸"。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

GEOMETRY_VERSION = "g5"  # 修改分析逻辑后递增，使缓存失效
VALUE_TOL = 0.05  # mm；图纸一般标到 0.1 mm 或整数
ANGLE_TOL = 0.5   # 度
DRILL_POINT_HALF_ANGLE = (55.0, 62.0)  # 118°/120° 钻尖
CHAMFER_HALF_ANGLE = (44.0, 46.0)
# 粗牙普通螺纹的底孔直径：这类盲孔是螺纹孔，一般不需要剖视表达
TAP_DRILL_DIAMETERS = (1.6, 2.05, 2.5, 3.3, 4.2, 5.0, 6.8, 8.5, 10.2, 12.0, 14.0, 17.5, 21.0)


@dataclass
class Hole:
    diameter: float
    depth: float | None       # 通孔为 None；含入口倒角的轴向深度
    through: bool
    count: int
    stepped: bool = False                                  # 沉孔/台阶孔（同轴还有更小的孔）
    # 能给这组孔定位的真实尺寸，按方向分组（"x"/"y"/"z"/"斜"）：孔轴到平面的距离、孔轴之间的分量
    locations: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class StepGeometry:
    bbox: list[float]
    linear: list[float]       # 所有可能出现在图纸上的线性尺寸（去重、排序）
    diameters: list[float]
    radii: list[float]
    angles: list[float]
    holes: list[Hole]
    chamfers: list[float]     # 45° 倒角的尺寸（轴向）
    chamfer_counts: dict[str, int] = field(default_factory=dict)  # 倒角尺寸 -> 个数
    outer_diameters: list[float] = field(default_factory=list)   # 轴、凸台等外圆直径
    fillet_radii: list[float] = field(default_factory=list)      # 圆角半径（不满一圈的圆柱面）
    needs_section: bool = False                                  # 有沉孔/非螺纹盲孔等内部结构，需要剖视图
    # 沿 x/y/z 每个方向上需要被尺寸确定的位置：垂直于该方向的平面位置、孔轴在该方向上的坐标
    positions: dict[str, list[float]] = field(default_factory=dict)
    source: str = ""
    notes: list[str] = field(default_factory=list)

    # ---------- 核对 ----------
    @staticmethod
    def _near(v: float, values: list[float], tol: float = VALUE_TOL) -> bool:
        return any(abs(v - x) <= tol for x in values)

    def has_linear(self, v: float) -> bool:
        return self._near(v, self.linear)

    def has_diameter(self, v: float) -> bool:
        return self._near(v, self.diameters)

    def has_radius(self, v: float) -> bool:
        return self._near(v, self.radii)

    def has_angle(self, v: float) -> bool:
        return self._near(v, self.angles, ANGLE_TOL) or self._near(180 - v, self.angles, ANGLE_TOL)

    def has_chamfer(self, size: float) -> bool:
        return self._near(size, self.chamfers)

    def holes_with_diameter(self, d: float) -> list[Hole]:
        return [h for h in self.holes if abs(h.diameter - d) <= VALUE_TOL]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "StepGeometry":
        d = json.loads(text)
        d["holes"] = [Hole(**h) for h in d["holes"]]
        return cls(**d)


def _uniq(values, nd: int = 3) -> list[float]:
    return sorted({round(v, nd) for v in values if v > VALUE_TOL})


def analyze_step(path: str | Path) -> StepGeometry:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.gp import gp_Vec
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    if int(reader.ReadFile(str(path))) != 1:  # IFSelect_RetDone
        raise RuntimeError(f"无法读取 STEP 文件：{path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []  # 文件里可能有多个实体，点的内外判断要对每个实体做
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        solids.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    if not solids:
        raise RuntimeError(f"STEP 中没有实体：{path}")

    def inside(p) -> bool:
        return any(BRepClass3d_SolidClassifier(sol, gp_Pnt(*p), 1e-6).State() == TopAbs_IN for sol in solids)

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    bbox = [x1 - x0, y1 - y0, z1 - z0]
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b))

    def sub(a, b):
        return tuple(p - q for p, q in zip(a, b))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def norm(a):
        return math.sqrt(dot(a, a))

    planes, cylinders, cones = [], [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        s = BRepAdaptor_Surface(face)
        t = s.GetType()
        if t == GeomAbs_Plane:
            p = s.Plane()
            n, o = p.Axis().Direction(), p.Location()
            planes.append(((n.X(), n.Y(), n.Z()), (o.X(), o.Y(), o.Z())))
        elif t == GeomAbs_Cylinder:
            c = s.Cylinder()
            a = c.Axis()
            d, o = a.Direction(), a.Location()
            v0, v1 = s.FirstVParameter(), s.LastVParameter()
            u0, u1 = s.FirstUParameter(), s.LastUParameter()
            # 凹凸判断：取面中点的外法线（指向材料外侧）。孔的外法线指向轴心，轴/凸台的外法线背离轴心
            pnt, nrm = gp_Pnt(), gp_Vec()
            BRepGProp_Face(face).Normal((u0 + u1) / 2, (v0 + v1) / 2, pnt, nrm)
            org = (o.X(), o.Y(), o.Z())
            dirv = (d.X(), d.Y(), d.Z())
            pv = (pnt.X() - org[0], pnt.Y() - org[1], pnt.Z() - org[2])
            radial = tuple(pv[i] - sum(pv[k] * dirv[k] for k in range(3)) * dirv[i] for i in range(3))
            concave = sum(radial[i] * (nrm.X(), nrm.Y(), nrm.Z())[i] for i in range(3)) < 0
            cylinders.append({"r": c.Radius(), "dir": dirv, "org": org,
                              "v": (min(v0, v1), max(v0, v1)), "u_span": abs(u1 - u0), "concave": concave})
        elif t == GeomAbs_Cone:
            c = s.Cone()
            a = c.Axis()
            d, o = a.Direction(), a.Location()
            cones.append({"r": c.RefRadius(), "half": math.degrees(abs(c.SemiAngle())),
                          "dir": (d.X(), d.Y(), d.Z()), "org": (o.X(), o.Y(), o.Z()),
                          "v": (s.FirstVParameter(), s.LastVParameter()),
                          "u_span": abs(s.LastUParameter() - s.FirstUParameter())})
        ex.Next()

    linear: list[float] = list(bbox)

    # 平行平面间距
    for (n1, o1), (n2, o2) in itertools.combinations(planes, 2):
        if abs(abs(dot(n1, n2)) - 1) < 1e-6:
            linear.append(abs(dot(n1, sub(o1, o2))))

    # 直线边：长度及在三个主方向上的投影
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        cur = BRepAdaptor_Curve(TopoDS.Edge_s(ex.Current()))
        if cur.GetType() == GeomAbs_Line:
            p0, p1 = cur.Value(cur.FirstParameter()), cur.Value(cur.LastParameter())
            vec = (p1.X() - p0.X(), p1.Y() - p0.Y(), p1.Z() - p0.Z())
            linear.append(norm(vec))
            linear += [abs(dot(vec, ax)) for ax in axes]
        ex.Next()

    # 孔：同轴（方向平行且轴线重合）、同半径的圆柱面合并
    def same_line(c1, c2):
        if abs(abs(dot(c1["dir"], c2["dir"])) - 1) > 1e-6:
            return False
        return norm(cross(sub(c2["org"], c1["org"]), c1["dir"])) < 1e-4

    def axial_extent(item, ref):
        """把面的 v 参数范围换算成参考轴上的坐标区间。"""
        base = dot(sub(item["org"], ref["org"]), ref["dir"])
        sign = 1 if dot(item["dir"], ref["dir"]) > 0 else -1
        lo, hi = item["v"]
        if "half" in item:  # 锥面的 v 是母线长度，换算为轴向长度
            k = math.cos(math.radians(item["half"]))
            lo, hi = lo * k, hi * k
        a, b = base + sign * lo, base + sign * hi
        return min(a, b), max(a, b)

    # 同半径、同轴线的圆柱面先归为一组，再按轴向区间切成互不相连的段：
    # 两面同轴的盲孔（BOTH SIDES）会被分成两段，各算一个孔
    lines: list[list[dict]] = []
    for c in cylinders:
        for g in lines:
            if abs(g[0]["r"] - c["r"]) < 1e-6 and same_line(g[0], c):
                g.append(c)
                break
        else:
            lines.append([c])

    def split_segments(group: list[dict]) -> list[list[dict]]:
        ref = group[0]
        items = sorted(group, key=lambda c: axial_extent(c, ref)[0])
        segs, cur, cur_hi = [], [], None
        for c in items:
            lo, hi = axial_extent(c, ref)
            if cur and lo > cur_hi + 0.2:
                segs.append(cur)
                cur = []
            cur.append(c)
            cur_hi = hi if cur_hi is None or len(cur) == 1 else max(cur_hi, hi)
        segs.append(cur)
        return segs

    def axis_empty(ref) -> bool:
        """沿轴线在包围盒范围内取样，全部不在实体内部 -> 轴线贯穿零件（通孔）。"""
        span = max(bbox) * 1.5
        for i in range(-60, 61):
            t = span * i / 60
            p = tuple(o + t * d for o, d in zip(ref["org"], ref["dir"]))
            if not (x0 - 1 <= p[0] <= x1 + 1 and y0 - 1 <= p[1] <= y1 + 1 and z0 - 1 <= p[2] <= z1 + 1):
                continue
            if inside(p):
                return False
        return True

    raw_holes, fillet_radii, outer_diameters = [], [], []
    for g in lines:
        for seg in split_segments(g):
            ref = seg[0]
            # 圆角（如方孔的 R10 角）只有部分圆周；真正的孔/外圆各面合起来是完整一圈
            if sum(c["u_span"] for c in seg) < 2 * math.pi - 0.01:
                fillet_radii.append(ref["r"])
                continue
            lo = min(axial_extent(c, ref)[0] for c in seg)
            hi = max(axial_extent(c, ref)[1] for c in seg)
            # 孔是凹面；轴、凸台的外圆柱面是凸面
            if sum(c["concave"] for c in seg) * 2 <= len(seg):
                outer_diameters.append(2 * ref["r"])
                continue
            for k in cones:  # 入口 45° 倒角并入孔深
                if CHAMFER_HALF_ANGLE[0] <= k["half"] <= CHAMFER_HALF_ANGLE[1] and same_line(ref, k):
                    klo, khi = axial_extent(k, ref)
                    if abs(klo - hi) < 0.2 or abs(khi - lo) < 0.2:
                        lo, hi = min(lo, klo), max(hi, khi)
            raw_holes.append({"d": 2 * ref["r"], "len": hi - lo, "ref": ref})

    # 孔位尺寸：孔轴之间沿主方向的分量、孔轴到平行平面的距离（斜向中心距图纸上几乎不标，不计入）
    axis_names = "xyz"

    def direction_of(n) -> str:
        i = max(range(3), key=lambda k: abs(n[k]))
        return axis_names[i] if abs(abs(n[i]) - 1) < 1e-6 else "斜"

    for h in raw_holes:
        o, d = h["ref"]["org"], h["ref"]["dir"]
        locs: dict[str, list[float]] = defaultdict(list)
        for n, po in planes:
            if abs(dot(n, d)) < 1e-6:  # 平面与孔轴平行：孔轴到平面的距离
                locs[direction_of(n)].append(abs(dot(n, sub(o, po))))
        for other in raw_holes:
            o2, d2 = other["ref"]["org"], other["ref"]["dir"]
            if other is not h and abs(abs(dot(d, d2)) - 1) < 1e-6:
                w = sub(o2, o)
                perp = sub(w, tuple(dot(w, d) * x for x in d))
                for i, ax in enumerate(axes):
                    if abs(dot(ax, d)) < 1e-6:
                        locs[axis_names[i]].append(abs(dot(perp, ax)))
        h["locations"] = {k: [v for v in vs if v > VALUE_TOL] for k, vs in locs.items()}
        linear += [v for vs in h["locations"].values() for v in vs]

    groups: dict[tuple, dict] = {}
    empty_cache: dict[int, bool] = {}
    for h in raw_holes:
        ref = h["ref"]
        empty = empty_cache.setdefault(id(ref), axis_empty(ref))
        stepped = any(o["d"] < h["d"] - 1e-6 and same_line(ref, o["ref"]) for o in raw_holes)
        through = empty and not stepped  # 同轴有更小的孔 -> 它是沉孔/台阶孔，不是通孔
        key = (round(h["d"], 3), None if through else round(h["len"], 3), through)
        g = groups.setdefault(key, {"count": 0, "stepped": stepped, "locations": defaultdict(set)})
        g["count"] += 1
        for k, vs in h["locations"].items():
            g["locations"][k] |= {round(v, 3) for v in vs}
        linear.append(h["len"])

    holes = [Hole(diameter=d, depth=depth, through=thr, count=g["count"], stepped=g["stepped"],
                  locations={k: sorted(vs) for k, vs in sorted(g["locations"].items()) if vs})
             for (d, depth, thr), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0))]
    needs_section = any(
        h.stepped or (not h.through and not any(abs(h.diameter - t) <= 0.06 for t in TAP_DRILL_DIAMETERS))
        for h in holes)

    angles = []
    for (n1, _), (n2, _) in itertools.combinations(planes, 2):
        a = math.degrees(math.acos(max(-1.0, min(1.0, abs(dot(n1, n2))))))
        if a > 0.5:
            angles += [a, 90 - a] if a < 90 - 0.5 else [a]

    positions: dict[str, set] = {k: set() for k in axis_names}
    for n, po in planes:
        k = direction_of(n)
        if k != "斜":
            positions[k].add(round(po["xyz".index(k)], 3))
    for h in raw_holes:
        o, d = h["ref"]["org"], h["ref"]["dir"]
        for i, k in enumerate(axis_names):
            if abs(d[i]) < 1e-6:  # 孔轴垂直于该方向 -> 孔在该方向上有一个位置
                positions[k].add(round(o[i], 3))

    chamfer_cones = [k for k in cones if CHAMFER_HALF_ANGLE[0] <= k["half"] <= CHAMFER_HALF_ANGLE[1]]
    chamfers = [abs(k["v"][1] - k["v"][0]) * math.cos(math.radians(k["half"])) for k in chamfer_cones]
    turns: dict[str, float] = defaultdict(float)  # 一个倒角可能被拆成几个面，按圆周角累加计数
    for k, size in zip(chamfer_cones, chamfers):
        turns[f"{round(size, 3):g}"] += k["u_span"] / (2 * math.pi)

    return StepGeometry(
        bbox=[round(b, 3) for b in bbox],
        linear=_uniq(linear),
        diameters=_uniq(2 * c["r"] for c in cylinders),
        radii=_uniq(c["r"] for c in cylinders),
        angles=_uniq(angles, 2),
        holes=holes,
        chamfers=_uniq(chamfers),
        chamfer_counts={k: round(v) for k, v in turns.items() if round(v) > 0},
        outer_diameters=_uniq(outer_diameters),
        fillet_radii=_uniq(fillet_radii),
        needs_section=needs_section,
        positions={k: sorted(v) for k, v in positions.items() if v},
        source=Path(path).name,
    )


def load_step_geometry(path: str | Path, cache_dir: Path | None = None) -> StepGeometry:
    """带缓存：同一个 STEP 只分析一次。"""
    p = Path(path)
    if cache_dir is not None:
        key = GEOMETRY_VERSION + "-" + hashlib.sha256(p.read_bytes()).hexdigest()[:24]
        cached = cache_dir / "geometry" / f"{key}.json"
        if cached.exists():
            return StepGeometry.from_json(cached.read_text(encoding="utf-8"))
        geo = analyze_step(p)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(geo.to_json(), encoding="utf-8")
        return geo
    return analyze_step(p)


# ---------- 用 STEP 几何给图纸打分 ----------

def verify_dimension(geo: StepGeometry, kind: str, value: float) -> bool:
    return {"linear": geo.has_linear, "depth": geo.has_linear, "diameter": geo.has_diameter,
            "radius": geo.has_radius, "angle": geo.has_angle}[kind](value)


def verify_hole(geo: StepGeometry, h) -> tuple[float, str]:
    """返回 (0..1, 说明)：直径必须真实存在；深度/通孔、数量与 STEP 一致再加分。"""
    real = geo.holes_with_diameter(h.diameter)
    if not real:
        return 0.0, f"STEP 中没有 Ø{h.diameter:g} 的孔"
    if h.through:
        depth_ok = any(r.through for r in real)
    elif h.depth is not None:
        depth_ok = any(r.depth is not None and abs(r.depth - h.depth) <= VALUE_TOL for r in real)
    else:
        depth_ok = True  # 没标深度：深度是否标全由覆盖率负责，这里不算错
    match = [r for r in real if (r.through if h.through else (h.depth is None or (r.depth is not None and abs(r.depth - h.depth) <= VALUE_TOL)))]
    counts = {r.count for r in (match or real)}
    count_ok = any(h.count in (c, c // 2) or (h.count == 1) for c in counts)
    score = 0.5 + 0.3 * depth_ok + 0.2 * count_ok
    notes = []
    if not depth_ok:
        notes.append("深度/通孔与 STEP 不符，实际为 " + ", ".join("THRU" if r.through else f"↧{r.depth:g}" for r in real))
    if not count_ok:
        notes.append(f"数量与 STEP 不符，实际 {sorted(counts)}")
    return score, "；".join(notes) or "与 STEP 一致"


def score_accuracy(geo: StepGeometry, cand):
    """准确性：图上每个数值（尺寸、孔、倒角）是否在 STEP 中真实存在。标错的扣分。"""
    from .score import CategoryResult

    res = CategoryResult(name="accuracy", score=0.0, weight=1)
    prec = []
    for x in cand.dimensions:
        ok = verify_dimension(geo, x.kind, x.value)
        prec.append(float(ok))
        tag = f"[{x.view}] {x.kind} {x.value:g}"
        res.add_item(tag, float(ok), "STEP 中存在" if ok else "STEP 中找不到这个尺寸（标错）")
        (res.matched if ok else res.missing).append(tag + ("" if ok else "：STEP 中找不到（标错）"))
    for h in cand.holes:
        sc, why = verify_hole(geo, h)
        prec.append(sc)
        tag = f"[{h.view}] {h.count}×Ø{h.diameter:g}" + (" THRU" if h.through else (f" ↧{h.depth:g}" if h.depth else ""))
        res.add_item(f"hole:{tag}", sc, why)
        (res.matched if sc >= 0.999 else res.missing).append(f"{tag}：{why}")
    for c in cand.chamfers:
        ok = geo.has_chamfer(c.size)
        prec.append(float(ok))
        res.add_item(f"chamfer:{c.size:g}", float(ok), "STEP 中存在" if ok else "STEP 中没有这个倒角")
        if not ok:
            res.missing.append(f"倒角 {c.size:g}：STEP 中没有")
    res.score = sum(prec) / len(prec) if prec else 0.0
    res.notes = [f"图上 {len(prec)} 个数值中，{sum(p >= 0.999 for p in prec)} 个与 STEP 完全一致"]
    return res


def _chain_determined(nodes: list[float], values: list[float]) -> tuple[int, int]:
    """沿一个方向：从零件一端出发，图上的尺寸若等于两个位置之差就把它们连起来，能连到的位置算"已确定"。

    约定：零件一端为基准；标了总尺寸则另一端也确定；关于中心对称的两个位置之间有尺寸时，两者都确定（对称标注）。
    """
    if len(nodes) < 2:
        return len(nodes), len(nodes)
    lo, hi = nodes[0], nodes[-1]
    mid = (lo + hi) / 2
    match = lambda gap: any(abs(v - gap) <= VALUE_TOL for v in values)
    edges = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:] if match(b - a)]
    det = {lo}
    if match(hi - lo):
        det.add(hi)
    for a, b in edges:
        if abs(a + b - 2 * mid) <= VALUE_TOL:
            det |= {a, b}
    changed = True
    while changed:
        changed = False
        for a, b in edges:
            if (a in det) != (b in det):
                det |= {a, b}
                changed = True
    return len(det), len(nodes)


def score_completeness(geo: StepGeometry, cand):
    """完整性：按统一规则从 STEP 推出必须标注的项目，逐项检查图上是否标出。

    规则（对所有零件相同）：
      外形尺寸   包围盒三个方向的总尺寸
      每组孔     孔径 0.35 + 深度或 THRU 0.35 + 数量 0.3
      位置链     沿 x/y/z，每个平面位置和孔中心位置都能从零件边缘经图上尺寸一路推出（不缺定位尺寸）
    得分 = 特征项平均 × 50% + 位置链平均 × 50%（定位是否完整和特征是否标出同等重要）
      外圆       轴、凸台的直径
      圆角       圆角半径
      倒角       尺寸 0.6 + 个数标全 0.4
      剖视图     有沉孔、非螺纹盲孔等内部结构时，至少一个剖视图
    """
    from .score import MIRROR_NOTE, CategoryResult, _dims_to_holes

    res = CategoryResult(name="completeness", score=0.0, weight=1)
    near = lambda a, b: a is not None and b is not None and abs(a - b) <= VALUE_TOL
    dims, callouts = _dims_to_holes(cand)
    lin = [x.value for x in dims if x.kind in ("linear", "depth")]
    mirror = any(MIRROR_NOTE.search(t) for t in cand.tech_requirements) or any(h.both_sides for h in cand.holes)
    items: list[float] = []      # 特征项
    pos_items: list[float] = []  # 位置链

    def add(item_id: str, sc: float, ok_text: str, miss_text: str, bucket: list | None = None) -> None:
        (items if bucket is None else bucket).append(sc)
        res.add_item(item_id, sc, ok_text if sc >= 0.999 else miss_text)
        (res.matched if sc >= 0.999 else res.missing).append(f"{item_id}：{ok_text if sc >= 0.999 else miss_text}")

    for b in geo.bbox:
        ok = any(near(v, b) for v in lin)
        add(f"外形尺寸 {b:g}", float(ok), "已标注", "未标注")

    for g in geo.holes:
        label = f"{g.count}×Ø{g.diameter:g}" + (" THRU" if g.through else f" ↧{g.depth:g}")
        same_d = [c for c in callouts if near(c.diameter, g.diameter)]
        if g.through:
            matching = [c for c in same_d if c.through or c.depth is None]
            depth_ok = any(c.through for c in matching)
        else:
            matching = [c for c in same_d if c.depth is None or near(c.depth, g.depth)]
            depth_ok = any(near(c.depth, g.depth) for c in same_d) or any(near(v, g.depth) for v in lin)
        count_ok = (any(c.count == g.count for c in matching)
                    or (mirror and any(2 * c.count == g.count for c in matching))
                    or sum(c.count for c in matching) == g.count)
        sc = 0.35 * bool(same_d) + (0.35 * depth_ok + 0.3 * count_ok if same_d else 0.0)
        miss = []
        if not same_d:
            miss.append("孔径未标")
        else:
            if not depth_ok:
                miss.append("未写 THRU" if g.through else "未标孔深")
            if not count_ok:
                miss.append("数量未标全")
        add(f"孔 {label}", sc, "已完整标注", "、".join(miss))

    diam_vals = [x.value for x in cand.dimensions if x.kind == "diameter"]
    for d in geo.outer_diameters:
        add(f"外圆 Ø{d:g}", float(any(near(v, d) for v in diam_vals)), "已标注", "未标直径")

    rad_vals = [x.value for x in cand.dimensions if x.kind == "radius"]
    for r in geo.fillet_radii:
        add(f"圆角 R{r:g}", float(any(near(v, r) for v in rad_vals)), "已标注", "未标半径")

    for size, n in geo.chamfer_counts.items():
        size_f = float(size)
        calls = [c for c in cand.chamfers if near(c.size, size_f)]
        total = sum(c.count for c in calls)
        sc = 0.6 * bool(calls) + 0.4 * (bool(calls) and total >= n)
        add(f"倒角 {size}×45°（共 {n} 处）", sc, "已完整标注",
            "未标注" if not calls else f"只标了 {total} 处")

    for axis, nodes in geo.positions.items():
        det, total = _chain_determined(nodes, lin)
        add(f"{axis} 方向位置（{total} 个）", det / total, "全部能由尺寸确定",
            f"只有 {det}/{total} 个位置能由尺寸确定（缺定位或定形尺寸）", pos_items)

    if geo.needs_section:
        ok = any(v.kind == "section" for v in cand.views)
        add("剖视图（零件有沉孔/盲孔等内部结构）", float(ok), "已有", "缺少剖视图")

    feat = sum(items) / len(items) if items else 1.0
    pos = sum(pos_items) / len(pos_items) if pos_items else 1.0
    res.score = 0.5 * feat + 0.5 * pos
    res.notes = [f"STEP 推出 {len(items)} 项必须标注的特征（{feat:.0%}）和 {len(pos_items)} 个方向的位置链（{pos:.0%}）"]
    return res
