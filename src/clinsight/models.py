"""领域模型。

数据流：Interview → PainPoint → Requirement → Analysis（交付物）
每一层都保留指回上一层的 id，保证"结论可溯源"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Utterance:
    idx: int
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, idx: int, raw: dict[str, Any]) -> "Utterance":
        return cls(idx=idx, speaker=str(raw.get("speaker", "未知")), text=str(raw.get("text", "")).strip())


@dataclass
class Interview:
    id: str
    role: str
    department: str
    setting: str          # 门诊 / 病房 / 科室管理 ...
    date: str
    duration_min: int
    transcript: list[Utterance] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Interview":
        return cls(
            id=str(raw["id"]),
            role=str(raw.get("role", "一线使用者")),
            department=str(raw.get("department", "")),
            setting=str(raw.get("setting", "")),
            date=str(raw.get("date", "")),
            duration_min=int(raw.get("duration_min", 0)),
            transcript=[Utterance.from_dict(i, u) for i, u in enumerate(raw.get("transcript", []), start=1)],
            notes=str(raw.get("notes", "")),
        )

    @property
    def title(self) -> str:
        return f"{self.department} · {self.role}（{self.setting}）"

    def turn_text(self, idx: int) -> str:
        for u in self.transcript:
            if u.idx == idx:
                return u.text
        return ""


@dataclass
class PainPoint:
    """一条被抽出来的隐性痛点，必须带原话证据。"""

    id: str
    role: str
    scene: str
    category: str
    quote: str
    turn: int
    phenomenon: str
    frequency: str
    frequency_score: int
    severity: int
    impact: str
    evidence_count: int = 1
    metrics: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Requirement:
    """一条可验证的需求条目：需求陈述 + 证据 + 验收信号 + 排序结果。"""

    id: str
    statement: str
    scene: str
    category: str
    roles: list[str]
    pain_ids: list[str]
    evidence: list[dict[str, Any]]
    acceptance: list[str]
    ai_pattern: str
    factors: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    moscow: str = ""
    rationale: list[str] = field(default_factory=list)
    compliance_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceFinding:
    id: str
    title: str
    status: str        # pass / fail / unknown
    detail: str
    basis: str         # 依据来源
    action: str        # 需要补的动作

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Analysis:
    interview_id: str
    interview_title: str
    mode: str
    pain_points: list[PainPoint]
    requirements: list[Requirement]
    feasibility: dict[str, Any]
    compliance: dict[str, Any]
    priority_check: dict[str, Any] = field(default_factory=dict)
    prd_markdown: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interview_id": self.interview_id,
            "interview_title": self.interview_title,
            "mode": self.mode,
            "pain_points": [p.to_dict() for p in self.pain_points],
            "requirements": [r.to_dict() for r in self.requirements],
            "feasibility": self.feasibility,
            "compliance": self.compliance,
            "priority_check": self.priority_check,
            "prd_markdown": self.prd_markdown,
            "stats": self.stats,
        }
