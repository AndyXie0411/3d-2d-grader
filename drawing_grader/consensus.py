"""多次采样的合并（借鉴 Harbor RewardKit 的 samples + agreement）。

同一张图纸让模型独立读 k 次，再逐项投票：
- 列表字段（尺寸、孔、视图……）：按"身份键"（如 视图+类型+数值）计数，每个身份键取各次出现次数的中位数，
  属性有分歧时取出现最多的那个版本。偶发的漏读/多读会被投票过滤掉。
- 标量字段（标题栏各栏、是否工程图）：多数投票。
- agreement：所有身份键里，k 次读数完全一致的比例。低于阈值说明这张图读得不稳定，应人工复核。
"""
from __future__ import annotations

import re
from collections import Counter
from statistics import median_low
from typing import Callable

from .schema import Criterion, Drawing, FeatureCheck, FidelityJudgement, TitleBlock

REVIEW_THRESHOLD = 0.8


def _nv(label: str) -> str:
    s = label.upper()
    for word in ("SECTION", "VIEW", "DETAIL"):
        s = s.replace(word, "")
    return re.sub(r"[^A-Z0-9]", "", s)


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def _nt(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


IDENTITY: dict[str, Callable] = {
    "views": lambda v: (v.kind, _nv(v.label)),
    "dimensions": lambda d: (_nv(d.view), d.kind, _r(d.value)),
    "holes": lambda h: (_nv(h.view), _r(h.diameter), h.count),
    "chamfers": lambda c: (_nv(c.view), _r(c.size), _r(c.angle)),
    "gdt": lambda g: (_nv(g.view), g.symbol, _r(g.value)),
    "roughness": lambda r: (_nv(r.view), _r(r.ra)),
    "datums": lambda x: x.upper(),
    "tech_requirements": _nt,
}


def _dump(x) -> str:
    return x.model_dump_json() if hasattr(x, "model_dump_json") else repr(x)


def _merge_field(name: str, samples: list[list]) -> tuple[list, int, int, list[str]]:
    key = IDENTITY[name]
    groups_per_sample = []
    for items in samples:
        g: dict = {}
        for it in items:
            g.setdefault(key(it), []).append(it)
        groups_per_sample.append(g)

    all_keys = []
    for g in groups_per_sample:
        for k in g:
            if k not in all_keys:
                all_keys.append(k)

    merged, stable, disagreements = [], 0, []
    for k in all_keys:
        counts = [len(g.get(k, [])) for g in groups_per_sample]
        n = median_low(counts)
        if len(set(counts)) == 1:
            stable += 1
        else:
            disagreements.append(f"{name} {k}: 各次读到 {counts} 个，取 {n}")
        if n == 0:
            continue
        versions = Counter(_dump(it) for g in groups_per_sample for it in g.get(k, []))
        lookup = {_dump(it): it for g in groups_per_sample for it in g.get(k, [])}
        ranked = [lookup[v] for v, _ in versions.most_common()]
        if len(versions) > 1 and len(set(counts)) == 1:
            disagreements.append(f"{name} {k}: 属性读法不一致，取出现最多的版本")
        merged += (ranked * n)[:n]
    return merged, stable, len(all_keys), disagreements


def _vote(values: list):
    return Counter(values).most_common(1)[0][0]


def merge_drawings(samples: list[Drawing]) -> tuple[Drawing, float, list[str]]:
    if len(samples) == 1:
        return samples[0], 1.0, []

    out = samples[0].model_copy(deep=True)
    stable = total = 0
    notes: list[str] = []
    for name in IDENTITY:
        merged, s, t, d = _merge_field(name, [getattr(x, name) for x in samples])
        setattr(out, name, merged)
        stable, total, notes = stable + s, total + t, notes + d

    # 标量：多数投票
    tb = {}
    for f in TitleBlock.model_fields:
        vals = [getattr(x.title_block, f) for x in samples]
        winner = _vote([_nt(v) if isinstance(v, str) else v for v in vals])
        tb[f] = next(v for v in vals if (_nt(v) if isinstance(v, str) else v) == winner)
        total += 1
        stable += len(set(_nt(v) if isinstance(v, str) else v for v in vals)) == 1
    out.title_block = TitleBlock(**tb)
    out.is_engineering_drawing = _vote([x.is_engineering_drawing for x in samples])

    # 防注入：多数样本看到才算；排版缺陷是自由文本，取缺陷条数为中位数的那次
    seen = [x for x in samples if x.injection_text]
    out.injection_text = seen[0].injection_text if len(seen) * 2 > len(samples) else []
    by_layout = sorted(samples, key=lambda x: len(x.layout_issues))
    out.layout_issues = by_layout[(len(by_layout) - 1) // 2].layout_issues

    return out, (stable / total if total else 1.0), notes


STATUS_ORDER = ["missing", "wrong", "shown_not_dimensioned", "shown_and_dimensioned"]


def merge_judgements(samples: list[FidelityJudgement]) -> tuple[FidelityJudgement, float]:
    if len(samples) == 1:
        return samples[0], 1.0
    ids = []
    for j in samples:
        for c in j.feature_checks:
            if c.feature_id not in ids:
                ids.append(c.feature_id)
    checks, stable = [], 0
    for fid in ids:
        found = [c for j in samples for c in j.feature_checks if c.feature_id == fid]
        statuses = [c.status for c in found] + ["missing"] * (len(samples) - len(found))
        # 中位数：按好坏排序后取中间
        med = sorted(statuses, key=STATUS_ORDER.index)[(len(statuses) - 1) // 2]
        stable += len(set(statuses)) == 1
        note = next((c.note for c in found if c.status == med), "")
        checks.append(FeatureCheck(feature_id=fid, status=med, note=note))

    def crit(name: str) -> Criterion:
        cs = [getattr(j, name) for j in samples]
        passed = sum(c.passed for c in cs) * 2 > len(cs)
        return Criterion(passed=passed, note=next(c.note for c in cs if c.passed == passed))

    names = ["outline_matches", "projection_correct", "sections_meaningful", "proportions_consistent"]
    stable += sum(len({getattr(j, n).passed for j in samples}) == 1 for n in names)
    by_ph = sorted(samples, key=lambda j: len(j.phantom_features))
    merged = FidelityJudgement(
        feature_checks=checks, **{n: crit(n) for n in names},
        phantom_features=by_ph[(len(by_ph) - 1) // 2].phantom_features,
    )
    total = len(ids) + len(names)
    return merged, (stable / total if total else 1.0)
