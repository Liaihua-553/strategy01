"""Boltz adapter placeholder and deployment notes for Stage04."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import PredictorResult


class BoltzAdapter:
    name = "boltz"

    def environment_status(self) -> dict[str, str]:
        return {"boltz_cli": shutil.which("boltz") or "MISSING"}

    def predict(self, target_pdb: Path, binder_sequence: str, out_dir: Path, sample_id: str, state_index: int) -> PredictorResult:
        if not shutil.which("boltz"):
            raise RuntimeError("Boltz CLI is not available in the current proteina-complexa environment.")
        raise NotImplementedError("Stage04 keeps Boltz in a separate benchmark/deployment step before making it the default predictor.")
