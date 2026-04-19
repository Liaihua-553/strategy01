"""AF2/ColabDesign adapter placeholder for Stage04.

This file intentionally does not call AF2 by default.  The remote Strategy01
environment has colabdesign installed, but AF2 parameter environment variables
were not set during Stage04 planning.  The implementation therefore exposes a
diagnostic method and a clear failure mode instead of silently producing fake
AF2 results.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import PredictorResult


class AF2ColabDesignAdapter:
    name = "af2_colabdesign"

    def environment_status(self) -> dict[str, str]:
        keys = ["AF2_DIR", "COLABDESIGN_DATA_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE"]
        return {key: ("<set>" if os.environ.get(key) else "MISSING") for key in keys}

    def predict(self, target_pdb: Path, binder_sequence: str, out_dir: Path, sample_id: str, state_index: int) -> PredictorResult:
        status = self.environment_status()
        if status.get("AF2_DIR") == "MISSING" and status.get("COLABDESIGN_DATA_DIR") == "MISSING":
            raise RuntimeError(
                "AF2/ColabDesign prediction requested, but AF2_DIR/COLABDESIGN_DATA_DIR is missing. "
                "Use the experimental_or_cached adapter for Stage04 smoke or configure AF2 parameters first."
            )
        raise NotImplementedError("Wire this adapter to the repository's existing binder evaluation runner after AF2 params are configured.")
