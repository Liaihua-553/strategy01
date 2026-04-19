"""Protenix adapter placeholder and deployment notes for Stage04."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .base import PredictorResult


class ProtenixAdapter:
    name = "protenix"

    def environment_status(self) -> dict[str, str]:
        return {"protenix_python_package": "FOUND" if importlib.util.find_spec("protenix") else "MISSING"}

    def predict(self, target_pdb: Path, binder_sequence: str, out_dir: Path, sample_id: str, state_index: int) -> PredictorResult:
        if not importlib.util.find_spec("protenix"):
            raise RuntimeError("Protenix is not installed in the current proteina-complexa environment.")
        raise NotImplementedError("Stage04 keeps Protenix in a separate benchmark/deployment step before making it the default predictor.")
