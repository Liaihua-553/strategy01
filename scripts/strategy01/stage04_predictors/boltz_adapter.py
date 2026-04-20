"""Boltz-2 predictor adapter used by Strategy01 Stage05.

The adapter is intentionally thin: it writes a Boltz YAML input, launches the
already deployed Boltz-2 CLI, and normalizes the output into the Stage04/Stage05
`PredictorResult` schema.  Heavy prediction outputs stay outside git.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import PredictorResult


DEFAULT_BOLTZ_CLI = Path("/data/kfliao/general_model/envs/boltz_cb04aec/bin/boltz")
DEFAULT_BOLTZ_CACHE = Path("/data/kfliao/general_model/boltz_cache")


@dataclass
class BoltzRunConfig:
    boltz_cli: Path = DEFAULT_BOLTZ_CLI
    cache_dir: Path = DEFAULT_BOLTZ_CACHE
    model: str = "boltz2"
    output_format: str = "pdb"
    recycling_steps: int = 1
    sampling_steps: int = 5
    diffusion_samples: int = 1
    devices: int = 1
    use_no_kernels: bool = True
    use_potentials: bool = False
    write_full_pae: bool = True
    write_full_pde: bool = True
    override: bool = True
    num_workers: int = 0
    max_parallel_samples: int | None = None


class BoltzAdapter:
    name = "boltz2"

    def __init__(self, config: BoltzRunConfig | None = None):
        self.config = config or BoltzRunConfig()

    def environment_status(self) -> dict[str, str]:
        cli = self.config.boltz_cli if self.config.boltz_cli.exists() else Path(shutil.which("boltz") or "")
        cache = self.config.cache_dir
        return {
            "boltz_cli": str(cli) if cli and cli.exists() else "MISSING",
            "cache_dir": str(cache),
            "boltz2_conf_ckpt": "FOUND" if (cache / "boltz2_conf.ckpt").exists() else "MISSING",
            "boltz2_aff_ckpt": "FOUND" if (cache / "boltz2_aff.ckpt").exists() else "MISSING",
            "mols_tar_or_dir": "FOUND" if (cache / "mols.tar").exists() or (cache / "mols").exists() else "MISSING",
            "model": self.config.model,
            "no_kernels": str(self.config.use_no_kernels),
        }

    @staticmethod
    def _safe_sequence(seq: str) -> str:
        seq = "".join(seq.split()).upper()
        if not seq:
            raise ValueError("Empty protein sequence is not valid for Boltz prediction.")
        allowed = set("ACDEFGHIKLMNPQRSTVWYX")
        bad = sorted(set(seq) - allowed)
        if bad:
            raise ValueError(f"Unsupported amino acid letters for Boltz input: {bad}")
        return seq.replace("X", "A")

    def write_input_yaml(
        self,
        yaml_path: Path,
        target_sequence: str,
        binder_sequence: str,
        *,
        target_template_pdb: Path | None = None,
        use_template: bool = True,
        force_template: bool = False,
        template_threshold: float = 2.0,
        template_id: str | None = None,
        msa: str = "empty",
        contact_constraints: list[tuple[int, int]] | None = None,
    ) -> None:
        """Write a minimal Boltz YAML for target chain A + binder chain B.

        Residue indices in optional contact constraints are 1-based within their
        respective chains.  For Stage05 smoke we keep constraints optional and
        primarily use the target-state PDB as a template.
        """
        target_sequence = self._safe_sequence(target_sequence)
        binder_sequence = self._safe_sequence(binder_sequence)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["version: 1", "sequences:"]
        for chain_id, seq in [("A", target_sequence), ("B", binder_sequence)]:
            lines.extend(
                [
                    "  - protein:",
                    f"      id: {chain_id}",
                    f"      sequence: {seq}",
                    f"      msa: {msa}",
                ]
            )
        if target_template_pdb is not None:
            lines.append("templates:")
            lines.append(f"  - pdb: {target_template_pdb}")
            lines.append("    chain_id: A")
            if template_id:
                lines.append(f"    template_id: {template_id}")
            lines.append(f"    force: {'true' if force_template else 'false'}")
            lines.append(f"    threshold: {template_threshold:.3f}")
        if contact_constraints:
            lines.append("constraints:")
            for target_idx, binder_idx in contact_constraints:
                lines.extend(
                    [
                        "  - contact:",
                        f"      token1: [A, {int(target_idx)}]",
                        f"      token2: [B, {int(binder_idx)}]",
                        "      max_distance: 8.0",
                        "      force: false",
                    ]
                )
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def build_command(self, yaml_path: Path, out_dir: Path) -> list[str]:
        cfg = self.config
        cli = cfg.boltz_cli if cfg.boltz_cli.exists() else Path(shutil.which("boltz") or "boltz")
        cmd = [
            str(cli),
            "predict",
            str(yaml_path),
            "--out_dir",
            str(out_dir),
            "--cache",
            str(cfg.cache_dir),
            "--model",
            cfg.model,
            "--output_format",
            cfg.output_format,
            "--devices",
            str(cfg.devices),
            "--accelerator",
            "gpu",
            "--recycling_steps",
            str(cfg.recycling_steps),
            "--sampling_steps",
            str(cfg.sampling_steps),
            "--diffusion_samples",
            str(cfg.diffusion_samples),
            "--num_workers",
            str(cfg.num_workers),
        ]
        if cfg.max_parallel_samples is not None:
            cmd.extend(["--max_parallel_samples", str(cfg.max_parallel_samples)])
        if cfg.use_no_kernels:
            cmd.append("--no_kernels")
        if cfg.use_potentials:
            cmd.append("--use_potentials")
        if cfg.write_full_pae:
            cmd.append("--write_full_pae")
        if cfg.write_full_pde:
            cmd.append("--write_full_pde")
        if cfg.override:
            cmd.append("--override")
        return cmd

    @staticmethod
    def prediction_dir(out_dir: Path, input_stem: str) -> Path:
        return out_dir / f"boltz_results_{input_stem}" / "predictions" / input_stem

    @staticmethod
    def _first_glob(path: Path, pattern: str) -> Path | None:
        matches = sorted(path.glob(pattern))
        return matches[0] if matches else None

    @staticmethod
    def _load_npz_array(path: Path | None, preferred_key: str | None = None) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        with np.load(path) as data:
            if preferred_key and preferred_key in data.files:
                return np.asarray(data[preferred_key])
            if not data.files:
                return None
            return np.asarray(data[data.files[0]])

    @staticmethod
    def _interchain_mean(matrix: np.ndarray | None, target_len: int | None, binder_len: int | None) -> float | None:
        if matrix is None or target_len is None or binder_len is None:
            return None
        if matrix.ndim != 2 or matrix.shape[0] < target_len + binder_len or matrix.shape[1] < target_len + binder_len:
            return None
        tb = matrix[:target_len, target_len : target_len + binder_len]
        bt = matrix[target_len : target_len + binder_len, :target_len]
        vals = np.concatenate([tb.reshape(-1), bt.reshape(-1)])
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else None

    @staticmethod
    def _binder_mean(values: np.ndarray | None, target_len: int | None, binder_len: int | None) -> float | None:
        if values is None or target_len is None or binder_len is None:
            return None
        flat = np.asarray(values).reshape(-1)
        if flat.shape[0] < target_len + binder_len:
            return None
        binder = flat[target_len : target_len + binder_len]
        binder = binder[np.isfinite(binder)]
        return float(binder.mean()) if binder.size else None

    def parse_existing_output(
        self,
        out_dir: Path,
        input_stem: str,
        *,
        sample_id: str,
        state_index: int,
        target_len: int | None = None,
        binder_len: int | None = None,
    ) -> PredictorResult:
        pred_dir = self.prediction_dir(out_dir, input_stem)
        pdb_path = self._first_glob(pred_dir, "*_model_0.pdb") or self._first_glob(pred_dir, "*.pdb")
        if pdb_path is None:
            raise FileNotFoundError(f"Boltz prediction PDB not found under {pred_dir}")
        confidence_path = self._first_glob(pred_dir, "confidence_*_model_0.json") or self._first_glob(pred_dir, "confidence_*.json")
        if confidence_path is None:
            raise FileNotFoundError(f"Boltz confidence JSON not found under {pred_dir}")
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        pae_path = self._first_glob(pred_dir, "pae_*_model_0.npz") or self._first_glob(pred_dir, "pae_*.npz")
        plddt_path = self._first_glob(pred_dir, "plddt_*_model_0.npz") or self._first_glob(pred_dir, "plddt_*.npz")
        pde_path = self._first_glob(pred_dir, "pde_*_model_0.npz") or self._first_glob(pred_dir, "pde_*.npz")
        pae = self._load_npz_array(pae_path, "pae")
        plddt = self._load_npz_array(plddt_path, "plddt")
        pde = self._load_npz_array(pde_path, "pde")
        interchain_pae = self._interchain_mean(pae, target_len, binder_len)
        interchain_pde = self._interchain_mean(pde, target_len, binder_len)
        binder_plddt = self._binder_mean(plddt, target_len, binder_len)
        complex_plddt = confidence.get("complex_plddt")
        protein_iptm = confidence.get("protein_iptm", confidence.get("iptm"))
        complex_iplddt = confidence.get("complex_iplddt", complex_plddt)
        if interchain_pae is not None:
            pae_factor = max(0.0, 1.0 - min(float(interchain_pae), 31.0) / 31.0)
        else:
            pae_factor = None
        if complex_iplddt is not None and protein_iptm is not None and pae_factor is not None:
            pdockq2_proxy = float(complex_iplddt) * float(protein_iptm) * pae_factor
        else:
            pdockq2_proxy = None
        metrics: dict[str, Any] = {
            "confidence_score": confidence.get("confidence_score"),
            "ptm": confidence.get("ptm"),
            "iptm": confidence.get("iptm"),
            "protein_iptm": protein_iptm,
            "complex_plddt_norm": complex_plddt,
            "complex_iplddt_norm": complex_iplddt,
            "binder_plddt_norm": binder_plddt,
            "interchain_pae_A": interchain_pae,
            "interchain_pde_A": interchain_pde,
            "pDockQ2_proxy": pdockq2_proxy,
            "pae_path": str(pae_path) if pae_path else None,
            "plddt_path": str(plddt_path) if plddt_path else None,
            "pde_path": str(pde_path) if pde_path else None,
            "prediction_dir": str(pred_dir),
        }
        return PredictorResult(
            sample_id=sample_id,
            state_index=state_index,
            complex_path=pdb_path,
            confidence_path=confidence_path,
            metrics=metrics,
            predictor_name=self.name,
            note="Parsed Boltz-2 prediction output.",
        )

    def predict(
        self,
        target_pdb: Path,
        binder_sequence: str,
        out_dir: Path,
        sample_id: str,
        state_index: int,
        *,
        target_sequence: str,
        target_template_pdb: Path | None = None,
        use_template: bool = True,
        force_template: bool = False,
        template_threshold: float = 2.0,
        template_id: str | None = None,
        msa: str = "empty",
        timeout_sec: int | None = None,
        target_len: int | None = None,
        binder_len: int | None = None,
    ) -> PredictorResult:
        status = self.environment_status()
        if status["boltz_cli"] == "MISSING":
            raise RuntimeError(f"Boltz CLI is not available. Status: {status}")
        out_dir.mkdir(parents=True, exist_ok=True)
        input_stem = f"{sample_id}_state{state_index:02d}_boltz"
        yaml_path = out_dir / "inputs" / f"{input_stem}.yaml"
        self.write_input_yaml(
            yaml_path,
            target_sequence,
            binder_sequence,
            target_template_pdb=(target_template_pdb or target_pdb) if use_template else None,
            force_template=force_template,
            template_threshold=template_threshold,
            template_id=template_id,
            msa=msa,
        )
        cmd = self.build_command(yaml_path, out_dir)
        log_path = out_dir / "logs" / f"{input_stem}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("BOLTZ_CACHE", str(self.config.cache_dir))
        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write("COMMAND: " + " ".join(cmd) + "\n")
            log_fh.flush()
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout_sec)
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Boltz prediction failed with exit code {proc.returncode}. Log tail:\n{tail}")
        result = self.parse_existing_output(
            out_dir,
            input_stem,
            sample_id=sample_id,
            state_index=state_index,
            target_len=target_len,
            binder_len=binder_len,
        )
        result.note = f"Boltz-2 prediction completed. input_yaml={yaml_path}; log={log_path}"
        return result



