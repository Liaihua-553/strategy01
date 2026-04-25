#!/usr/bin/env python
"""Stage07 Boltz2 MSA diagnostics and reusable MSA cache indexing.

This script does not require GPU by default.  It scans prior Boltz outputs/logs,
checks whether remote ColabFold MSA server was used, indexes produced per-chain
MSA CSVs by sequence hash when possible, and writes a compact report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter  # noqa: E402

DEFAULT_BOLTZ_OUTPUTS = REPO_ROOT / "data" / "strategy01" / "stage06_production" / "boltz_outputs"
DEFAULT_CACHE = REPO_ROOT / "data" / "strategy01" / "stage07_msa_cache"
DEFAULT_OUT = REPO_ROOT / "reports" / "strategy01" / "probes" / "stage07_msa_diagnostics_summary.json"


def scan_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    server_enabled = "MSA server enabled" in text or "api.colabfold.com" in text
    colabfold_hits = text.count("api.colabfold.com")
    seconds = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)", text, flags=re.I):
        try:
            seconds.append(float(match.group(1)))
        except ValueError:
            pass
    return {
        "path": str(path),
        "server_enabled": server_enabled,
        "colabfold_hits": colabfold_hits,
        "line_count": text.count("\n") + 1,
        "max_logged_seconds": max(seconds) if seconds else None,
        "has_error": any(token in text.lower() for token in ["traceback", "error", "failed", "exception"]),
    }


def index_msa_csvs(root: Path, cache_dir: Path) -> list[dict[str, Any]]:
    indexed = []
    for csv_path in root.rglob("msa/*_*.csv"):
        try:
            text = csv_path.read_text(encoding="utf-8", errors="replace")
            seq = None
            for line in text.splitlines():
                if line.startswith(">") or line.lower().startswith("key,"):
                    continue
                parts = line.split(",")
                candidate = parts[-1].strip().upper() if parts else ""
                if len(candidate) >= 10 and set(candidate) <= set("ACDEFGHIKLMNPQRSTVWYX-"):
                    seq = candidate.replace("-", "")
                    break
            if not seq:
                continue
            chain_label = "chain"
            out = BoltzAdapter.cache_msa_csv(csv_path, seq, cache_dir, chain_label)
            indexed.append({"source": str(csv_path), "cache": str(out), "sequence_hash": BoltzAdapter.sequence_hash(seq), "sequence_len": len(seq)})
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail on one malformed file.
            indexed.append({"source": str(csv_path), "error": str(exc)})
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz-outputs", type=Path, default=DEFAULT_BOLTZ_OUTPUTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-logs", type=int, default=200)
    args = parser.parse_args()
    logs = sorted(args.boltz_outputs.rglob("*.log"))[: args.max_logs]
    log_rows = [scan_log(path) for path in logs]
    msa_rows = index_msa_csvs(args.boltz_outputs, args.cache_dir)
    summary = {
        "boltz_outputs": str(args.boltz_outputs),
        "log_count_scanned": len(log_rows),
        "server_enabled_logs": sum(1 for row in log_rows if row["server_enabled"]),
        "error_logs": sum(1 for row in log_rows if row["has_error"]),
        "msa_cache_dir": str(args.cache_dir),
        "msa_csv_indexed": len([row for row in msa_rows if "cache" in row]),
        "logs_sample": log_rows[:20],
        "msa_sample": msa_rows[:20],
        "local_msa_fallback_note": "If server_enabled_logs is low or runtime is dominated by MSA, run colabfold_search/MMseqs2 on CPU nodes and pass cached CSV/A3M paths into Boltz YAML per chain.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()