"""Stage04 predictor adapter interfaces.

The adapters keep heavy external predictors outside the training loop.  Stage04
uses offline complex predictions or experimental complexes, then converts their
outputs into geometry labels and quality-proxy labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class PredictorResult:
    sample_id: str
    state_index: int
    complex_path: Path
    confidence_path: Path | None
    metrics: dict[str, Any]
    predictor_name: str
    note: str = ""


class ComplexPredictorAdapter(Protocol):
    name: str

    def predict(self, target_pdb: Path, binder_sequence: str, out_dir: Path, sample_id: str, state_index: int) -> PredictorResult:
        ...


class ExperimentalComplexAdapter:
    """Adapter for pre-existing experimental or already-predicted complexes."""

    name = "experimental_or_cached_complex"

    def predict(self, target_pdb: Path, binder_sequence: str, out_dir: Path, sample_id: str, state_index: int) -> PredictorResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        return PredictorResult(
            sample_id=sample_id,
            state_index=state_index,
            complex_path=target_pdb,
            confidence_path=None,
            metrics={"confidence_source": "geometry_proxy_no_external_predictor"},
            predictor_name=self.name,
            note="Using a cached/experimental complex file; no external predictor was called.",
        )
