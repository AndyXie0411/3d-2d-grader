"""总评分：把各个组成部分合成一个 0-100 分。

组成部分（缺少某种输入时，对应部分自动跳过，权重按剩余部分重新归一化）：
  accuracy     准确性（规则）                  需要 --step：图上每个数值是否在 STEP 中真实存在
  completeness 完整性（规则）                  需要 --step：按统一规则从 STEP 推出的必标项目是否都标了
  reference    与参考图纸逐项比对（可选）      需要 --ref（图片或 JSON）
  fidelity     与 3D 模型的特征一致性（VLM）  需要 3D 模型图片
  standard     BenchCAD 规格完整性（规则）     总是计算
  consistency  图纸内部自洽性（规则）          总是计算
  layout       排版质量                         总是计算

借鉴 Harbor / RewardKit 的几条机制：
  门槛（required-pass）  不是工程图 -> 0 分；没有正交视图、几乎没有尺寸 -> 总分封顶 GATE_CAP；
                         门槛不过时都跳过昂贵的 3D 核对
  防注入（guard=penalize）图上有试图影响评分的文字 -> 总分 0
  逐条明细               每个评分项都有 id/score/reason，写入 reward-details.json
  采样一致率             多次采样时报告 agreement，低于阈值标记 needs_review
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .consensus import REVIEW_THRESHOLD
from .rubric import check_consistency, check_standard
from .schema import Drawing, FeatureInventory, FidelityJudgement
from .score import CategoryResult, grade as grade_reference, score_layout

COMPONENT_WEIGHTS = {
    "accuracy": 35,
    "completeness": 40,
    "reference": 25,
    "fidelity": 30,
    "standard": 10,
    "consistency": 10,
    "layout": 5,
}

FEATURE_STATUS_SCORE = {"shown_and_dimensioned": 1.0, "shown_not_dimensioned": 0.5, "missing": 0.0, "wrong": 0.0}
PHANTOM_PENALTY = 0.05
GATE_CAP = 20.0
MIN_DIMENSIONS_FOR_GATE = 3


@dataclass
class Component:
    name: str
    score: float  # 0..100
    weight: float
    details: list[CategoryResult] = field(default_factory=list)


@dataclass
class CaseReport:
    total: float
    components: list[Component]
    gates: list[dict] = field(default_factory=list)
    capped_by: str = ""
    agreement: dict[str, float] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return any(v < REVIEW_THRESHOLD for v in self.agreement.values())

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "capped_by": self.capped_by,
            "needs_review": self.needs_review,
            "agreement": {k: round(v, 3) for k, v in self.agreement.items()},
            "gates": self.gates,
            "components": [
                {
                    "name": c.name,
                    "score": round(c.score, 1),
                    "weight": c.weight,
                    "details": [
                        {"name": d.name, "score": round(d.score * 100, 1), "weight": d.weight,
                         "matched": d.matched, "missing": d.missing, "extra": d.extra, "notes": d.notes}
                        for d in c.details
                    ],
                }
                for c in self.components
            ],
        }

    def reward(self) -> dict[str, float]:
        """Harbor 风格 reward.json：reward 为总分（0~1），其余为各部分得分；权重为 0 的部分不写。"""
        out = {"reward": round(float(self.total) / 100, 4)}
        out.update({c.name: round(float(c.score) / 100, 4) for c in self.components if c.weight > 0})
        return out

    def reward_details(self) -> dict:
        """RewardKit 风格 reward-details.json：每个评分项的 id / 分数 / 理由。"""
        return {
            "reward": round(float(self.total) / 100, 4),
            "capped_by": self.capped_by,
            "needs_review": self.needs_review,
            "agreement": {k: round(v, 3) for k, v in self.agreement.items()},
            "gates": self.gates,
            "criteria": [
                {**item, "component": c.name, "component_weight": c.weight}
                for c in self.components for d in c.details for item in d.items
            ],
        }


def check_gates(cand: Drawing) -> list[dict]:
    n_dims = len(cand.dimensions) + len(cand.holes)
    ortho = [v for v in cand.views if v.kind != "isometric"]
    return [
        {"id": "gate:is_engineering_drawing", "passed": cand.is_engineering_drawing,
         "reason": "" if cand.is_engineering_drawing else "不是机械工程图"},
        {"id": "gate:has_orthographic_view", "passed": bool(ortho),
         "reason": "" if ortho else "没有任何正交/剖视/向视图"},
        {"id": f"gate:at_least_{MIN_DIMENSIONS_FOR_GATE}_dimensions", "passed": n_dims >= MIN_DIMENSIONS_FOR_GATE,
         "reason": "" if n_dims >= MIN_DIMENSIONS_FOR_GATE else f"只有 {n_dims} 个尺寸/孔标注"},
    ]


def score_fidelity(inv: FeatureInventory, j: FidelityJudgement) -> CategoryResult:
    res = CategoryResult(name="fidelity", score=0.0, weight=1)
    by_id = {c.feature_id: c for c in j.feature_checks}
    feat_scores = []
    for f in inv.features:
        c = by_id.get(f.id)
        s = FEATURE_STATUS_SCORE[c.status] if c else 0.0
        feat_scores.append(s)
        line = f"{f.id} {f.description}" + (f"：{c.status}，{c.note}" if c else "：评审未覆盖，按缺失计")
        (res.matched if s >= 1 else res.missing).append(line)
        res.add_item(f"feature.{f.id}", s, f"{f.description} -> {c.status}：{c.note}" if c else "评审未覆盖，按缺失计")
    feat = sum(feat_scores) / len(feat_scores) if feat_scores else 1.0

    crit = {"outline_matches": ("外轮廓一致", j.outline_matches),
            "projection_correct": ("投影关系正确", j.projection_correct),
            "sections_meaningful": ("剖视合理", j.sections_meaningful),
            "proportions_consistent": ("比例与尺寸一致", j.proportions_consistent)}
    for key, (name, c) in crit.items():
        (res.matched if c.passed else res.missing).append(f"{name}：{c.note}")
        res.add_item(key, float(c.passed), c.note)
    crit_score = sum(c.passed for _, c in crit.values()) / len(crit)

    res.extra = list(j.phantom_features)
    for i, ph in enumerate(j.phantom_features, 1):
        res.add_item(f"phantom{i}", 0, f"图上多出 3D 模型中不存在的特征：{ph}")
    res.score = max(0.0, 0.6 * feat + 0.4 * crit_score - PHANTOM_PENALTY * len(j.phantom_features))
    res.notes = [f"特征覆盖 {feat:.0%}（权重 60%），通用标准 {crit_score:.0%}（权重 40%），"
                 f"每个凭空多出的特征扣 {PHANTOM_PENALTY:.0%}"]
    return res


def grade_case(
    cand: Drawing,
    ref: Drawing | None = None,
    fidelity: tuple[FeatureInventory, FidelityJudgement] | None = None,
    weights: dict[str, float] | None = None,
    agreement: dict[str, float] | None = None,
    geo=None,
) -> CaseReport:
    w = {**COMPONENT_WEIGHTS, **(weights or {})}
    comps: list[Component] = []

    if geo is not None:
        from .step_geometry import score_accuracy, score_completeness

        for name, fn in (("accuracy", score_accuracy), ("completeness", score_completeness)):
            r = fn(geo, cand)
            comps.append(Component(name, r.score * 100, w[name], [r]))
    if ref is not None:
        rep = grade_reference(ref, cand, geo=geo)
        comps.append(Component("reference", rep.total, w["reference"], rep.categories))
    if fidelity is not None:
        r = score_fidelity(*fidelity)
        comps.append(Component("fidelity", r.score * 100, w["fidelity"], [r]))
    for name, fn in (("standard", check_standard), ("consistency", check_consistency)):
        r = fn(cand)
        comps.append(Component(name, r.score * 100, w[name], [r]))
    r = score_layout(cand, 1)
    comps.append(Component("layout", r.score * 100, w["layout"], [r]))

    total_w = sum(c.weight for c in comps)
    total = sum(c.score * c.weight for c in comps) / total_w if total_w else 0.0

    gates = check_gates(cand)
    capped_by = ""
    if cand.injection_text:
        total, capped_by = 0.0, f"图上有试图影响评分的文字：{cand.injection_text}"
    elif not cand.is_engineering_drawing:
        # 空白图、渲染图等：图上没有可查的内容，规则项会"空洞地满分"，必须直接判 0
        total, capped_by = 0.0, "不是机械工程图，判 0 分"
    elif not all(g["passed"] for g in gates):
        failed = [g["id"] for g in gates if not g["passed"]]
        if total > GATE_CAP:
            total = GATE_CAP
        capped_by = f"未通过门槛 {failed}，总分封顶 {GATE_CAP:g}"
    return CaseReport(total=total, components=comps, gates=gates, capped_by=capped_by, agreement=agreement or {})


def grade_files(
    candidate: str | Path,
    model_images: list[str | Path] | None = None,
    reference: str | Path | None = None,
    weights: dict[str, float] | None = None,
    inventory: str | Path | None = None,
    samples: int = 1,
    judge_samples: int = 1,
    step: str | Path | None = None,
) -> tuple[CaseReport, Drawing]:
    """文件级入口：候选图纸可以是图片或 JSON；参考可以是图片或 JSON；3D 图片可选（可多张）。
    inventory：预先生成并人工校对过的 3D 特征清单 JSON；给出时不再调用模型生成清单。
    samples / judge_samples：读图 / 3D 核对的采样次数（>1 时投票合并并报告一致率）。
    step：给 AI 的 STEP 文件；提供时用真实几何核对图上的每个数值。"""
    from .extract import CACHE_DIR, extract_drawing, extract_drawing_sampled, inventory_3d, judge_fidelity_sampled

    geo = None
    if step:
        from .step_geometry import load_step_geometry

        geo = load_step_geometry(step, CACHE_DIR)

    agreement: dict[str, float] = {}
    cand_path = Path(candidate)
    if cand_path.suffix.lower() == ".json":
        cand = Drawing.model_validate_json(cand_path.read_text(encoding="utf-8"))
    else:
        cand, agreement["drawing"], _ = extract_drawing_sampled(cand_path, samples)

    ref = None
    if reference:
        rp = Path(reference)
        ref = (Drawing.model_validate_json(rp.read_text(encoding="utf-8")) if rp.suffix.lower() == ".json"
               else extract_drawing(rp))

    fid = None
    gates_ok = all(g["passed"] for g in check_gates(cand)) and not cand.injection_text
    if model_images and gates_ok:  # 门槛不过就不做最贵的 3D 核对，总分反正会被封顶
        if cand_path.suffix.lower() == ".json":
            raise ValueError("3D 一致性评审需要图纸原图，--cand 不能是 JSON")
        if inventory:
            inv = FeatureInventory.model_validate_json(Path(inventory).read_text(encoding="utf-8"))
        else:
            inv = inventory_3d(model_images)
        judgement, agreement["fidelity"] = judge_fidelity_sampled(model_images, inv, cand_path, judge_samples)
        fid = (inv, judgement)
    if samples <= 1:
        agreement.pop("drawing", None)
    if judge_samples <= 1:
        agreement.pop("fidelity", None)
    return grade_case(cand, ref, fid, weights, agreement, geo), cand
