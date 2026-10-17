"""【不需要参考图纸】的两类纯规则检查，适用于任意零件：

- standard：图纸是否具备能从 STEP 几何完成的出图要素（视图、轴测图、足够的尺寸）
- consistency：图纸内部是否自洽（基准、剖切符号、螺纹底孔等）——只检查图上已经写了的内容，没写不扣分

公平原则：STEP 中没有的信息（公差、形位公差、粗糙度、技术要求、标题栏内容）不要求出现。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Drawing
from .score import CategoryResult

# 粗牙普通螺纹的标准底孔直径（mm）
TAP_DRILL = {
    "M2": 1.6, "M2.5": 2.05, "M3": 2.5, "M4": 3.3, "M5": 4.2, "M6": 5.0, "M8": 6.8,
    "M10": 8.5, "M12": 10.2, "M14": 12.0, "M16": 14.0, "M20": 17.5, "M24": 21.0,
}

MIN_DIMENSIONS = 10


@dataclass
class Check:
    name: str
    passed: float  # 0..1
    detail: str = ""


def _result(name: str, weight: float, checks: list[Check]) -> CategoryResult:
    res = CategoryResult(name=name, score=0.0, weight=weight)
    applicable = [c for c in checks]
    if not applicable:
        res.score = 1.0
        return res
    res.score = sum(c.passed for c in applicable) / len(applicable)
    for c in applicable:
        line = c.name + (f"：{c.detail}" if c.detail else "")
        (res.matched if c.passed >= 0.999 else res.missing).append(line)
        res.add_item(c.name, c.passed, c.detail or ("通过" if c.passed >= 0.999 else "未通过"))
    return res


# ---------- 规格完整性 ----------

def check_standard(d: Drawing, weight: float = 1.0) -> CategoryResult:
    """只要求能从 STEP 几何完成的出图要素；形位公差、粗糙度、技术要求、标题栏内容等 STEP 中没有，不作要求。"""
    ortho = [v for v in d.views if v.kind != "isometric"]
    checks = [
        Check("有主视图", float(any(v.kind == "main" for v in d.views))),
        Check("至少 3 个正交/剖/向视图", min(1.0, len(ortho) / 3), f"实际 {len(ortho)} 个"),
        Check("有轴测图", float(any(v.kind == "isometric" for v in d.views))),
        Check(f"尺寸数量 ≥ {MIN_DIMENSIONS}（含孔标注）",
              min(1.0, (len(d.dimensions) + len(d.holes)) / MIN_DIMENSIONS),
              f"实际 {len(d.dimensions)} 个尺寸 + {len(d.holes)} 组孔标注"),
    ]
    return _result("standard", weight, checks)


# ---------- 内部一致性 ----------

def _view_letter(label: str) -> str | None:
    """'A-A' -> 'A'，'VIEW G' -> 'G'，'SECTION B-B' -> 'B'。"""
    s = label.upper().replace("SECTION", "").replace("VIEW", "").strip()
    m = re.fullmatch(r"([A-Z])\s*-\s*\1", s) or re.fullmatch(r"([A-Z])", s)
    return m.group(1) if m else None


def check_consistency(d: Drawing, weight: float = 1.0) -> CategoryResult:
    checks: list[Check] = []

    # 1. 形位公差引用的基准必须在图上标出
    datums = {x.upper() for x in d.datums}
    for g in d.gdt:
        undefined = [x for x in g.datums if x.upper() not in datums]
        checks.append(Check(f"[{g.view}] {g.symbol} 引用的基准已定义", float(not undefined),
                            f"未定义基准 {undefined}" if undefined else ""))
    # 2. 标出的基准应当被使用
    used = {x.upper() for g in d.gdt for x in g.datums}
    for x in sorted(datums):
        checks.append(Check(f"基准 {x} 被形位公差引用", float(x in used)))

    # 3. 剖切/向视符号与视图一一对应
    labelled = {_view_letter(v.label): v for v in d.views if v.kind in ("section", "direction", "detail")}
    labelled.pop(None, None)
    marked = {x.upper() for v in d.views for x in v.cut_lines_shown}
    for x in sorted(marked):
        checks.append(Check(f"剖切/向视符号 {x} 有对应视图", float(x in labelled)))
    for x, v in sorted(labelled.items()):
        checks.append(Check(f"视图 {v.label} 在其他视图上有剖切/投射符号", float(x in marked)))
    # 剖视图应画剖面线
    for v in d.views:
        if v.kind == "section":
            checks.append(Check(f"剖视图 {v.label} 画了剖面线", float(v.hatched)))

    # 4. 视图标签唯一
    labels = [v.label.upper() for v in d.views]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    checks.append(Check("视图标签不重复", float(not dup), f"重复 {dup}" if dup else ""))

    # 5. 孔标注合理性
    for h in d.holes:
        tag = f"[{h.view}] {h.count}×Ø{h.diameter:g}"
        if h.through and h.depth is not None:
            checks.append(Check(f"{tag} 通孔不应再标孔深", 0.0))
        if h.thread:
            key = re.match(r"M\d+(\.\d+)?", h.thread.upper())
            std = TAP_DRILL.get(key.group(0)) if key else None
            if std is not None:
                ok = abs(std - h.diameter) <= 0.11
                checks.append(Check(f"{tag} {h.thread} 底孔直径正确", float(ok),
                                    "" if ok else f"标准底孔 Ø{std:g}"))
            if h.depth is not None and h.thread_depth is not None:
                ok = h.thread_depth < h.depth
                checks.append(Check(f"{tag} 螺纹深度 < 钻孔深度", float(ok),
                                    "" if ok else f"螺纹 {h.thread_depth:g} ≥ 孔深 {h.depth:g}"))

    # 6. 粗糙度取值在常用系列内
    series = {0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5, 25, 50}
    for r in d.roughness:
        checks.append(Check(f"[{r.view}] Ra{r.ra:g} 属于标准系列", float(any(abs(r.ra - s) < 1e-6 for s in series))))

    return _result("consistency", weight, checks)
