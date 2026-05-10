# Strategy01 Stage19 exact benchmark bridge 报告

## 阶段目标

Stage18 解决了 target-only 候选的 severe clash hard-gate 与 bounded relief，但输出仍主要是 JSON proxy。要进入真正的 B0/B1/B2 exact benchmark，必须把 B1 target-only 采样得到的 state-specific binder 几何写成结构文件，再与 exact complex 的 contact map、interface RMSD、clash rate 做对齐评估。

本阶段先实现桥接能力：**可选输出每个 sample/seed/state 的 generated binder CA PDB**。这不是 legacy average，而是直接来自 `bb_ca_states[k]`，符合 Stage12/18 的科学契约。

## 代码改动

- `scripts/strategy01/stage12c_de_novo_smoke.py`
  - 新增 `write_binder_ca_pdb()`。
  - 新增 `--output-pdb-dir`。
  - 当启用时，在 `rollout_final()` 末尾按 sample/state 写出 `stateXX_binder_ca.pdb`。
  - 输出坐标从 Complexa 内部 nm 转成 PDB Å。
  - 明确只写 `x_states["bb_ca"][bi, si]`，即 state-specific generated pose；不写 weighted-average legacy 坐标。
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`
  - 新增 `--write-candidate-pdb-dir`。
  - 每个 candidate 的 JSON 行记录 `written_pdb_dirs`，方便后续 exact benchmark 找到对应结构。

## CPU probe

命令使用 CPU、1 sample、1 seed、1 step，只验证接口和 PDB 写出，不作为科学性能指标。

- probe summary：`{'first_seed_identity_mean': 0.04545454680919647, 'proxy_selected_identity_mean': 0.04545454680919647, 'oracle_identity_mean': 0.04545454680919647, 'proxy_matches_oracle_rate': 1.0, 'safe_candidate_available_rate': 1.0, 'selected_safe_rate': 1.0}`
- 输出目录：`reports/strategy01/probes/stage19_cpu_pdb_probe_pdbs`
- selected candidate PDB dirs：`['reports/strategy01/probes/stage19_cpu_pdb_probe_pdbs/sample_000_seed_1207/2ofq_A_B__v00__k3__hexact']`

已确认生成：

- `state00_binder_ca.pdb`
- `state01_binder_ca.pdb`
- `state02_binder_ca.pdb`

## warning / error 审查

- optional CA/residue feature warning：仍是 no-leak 设计预期。
- CCD/PDB mirror warning：本 probe 不调用 mirror 解析，不影响 PDB 写出。
- 无 Traceback/OOM/NaN。

## 当前意义

这一步没有宣称模型达到科学目标；它解决的是评估链路缺口。下一步可以在 GPU 可用时运行 Stage18 safe-select + PDB 输出，然后用 `stage08b_exact_geometry_benchmark.py` 的 contact/RMSD 函数或新 Stage19 exact evaluator，对 selected B1 state-specific PDB 与 exact complexes 做正式 B0/B1/B2 对比。

## 下一步

1. GPU 可用后，重跑 Stage18 iter8 val36，并启用 `--write-candidate-pdb-dir`。
2. 写 Stage19 exact evaluator：读取 per-sample selected candidate PDB、exact complex paths、target state paths，计算 contact-F1、direct/aligned binder CA RMSD、severe clash rate、contact persistence。
3. 若 Stage18 safe-select 在 exact metrics 上 contact/RMSD 仍差，则回到训练层加强 target-relative bb_ca flow，而不是继续只优化 sequence identity。
