# Strategy01 Stage12：Coupled Full Multistate Product-Space Flow 实施报告

## 阶段目标

Stage12 的目标是把 Strategy01 从“多状态 pose/latent repair”推进到真正的 target-only de novo multistate binder generation：只输入 K 个 target 构象、target 序列、hotspot/pocket/mask，模型从噪声生成 binder 的相对空间位置、backbone、local latent，以及一个共享 binder 序列。

本阶段的核心修正是：不能只让 K 个 state 独立生成 latent 后再投票出序列。这样容易退化成 K 个单状态 binder 的后处理平均。Stage12 必须把 shared sequence 作为生成过程内部的一等耦合信号，让它参与 K 条 state-specific flow 的去噪。

## 代码改动

1. `LocalLatentsTransformerMultistate`
   - 新增 `cross_state_shared_seq_attention`、`shared_seq_token_update`、`shared_seq_to_state_projector`、`shared_seq_token_head`。
   - 对同一 binder residue 的 K 个 state token 做跨状态 attention 和加权池化，得到 `shared_seq_tokens [B,Nb,D]`。
   - `shared_seq_tokens` 先产生 `seq_logits_shared_tokens [B,Nb,20]`，再反馈到每个 state 的结构/latent token。
   - 最终 `seq_logits_shared` 由 baseline shared head、Stage12 shared-token head、state-specific robust consensus 三部分共同决定。
   - 新增 `de_novo_multistate_mode` guard：如果输入包含 `init_bb_ca_states` 或 source-complex/source-interface 字段，直接报错。

2. `multistate_loss.py`
   - 新增 target-derived center helper，用 target/hotspot CA center 初始化 de novo `bb_ca` noise。
   - `de_novo_multistate` 模式下禁止 `init_bb_ca_states`，避免把目标退化为 inverse folding 或 pose repair。
   - `local_latents` corruption 使用 shared-main + small state-residual noise，使 K 条 latent trajectory 保持同一 shared-sequence 主随机变量，同时允许 state-specific 适配。
   - `x_t_states/t_states/x_0_states` 保持为主张量；weighted-average `x_t/x_1` 只为旧接口兼容。

3. Stage12 脚本
   - `scripts/strategy01/stage12_flow_coupling_probes.py`：CPU shape/no-leak/permutation/corruption probe。
   - `scripts/strategy01/stage12_de_novo_multistate_training.py`：Stage12 训练入口契约，固定 target-only 约束。
   - `scripts/strategy01/stage12_de_novo_multistate_sampling.py`：Stage12 sampling 契约，明确 full `bb_ca + local_latents` flow。
   - `scripts/strategy01/stage12_exact_benchmark.py`：B0/B1/B2 benchmark 口径契约。

## 科学意义

原始 Complexa 的生成变量是 `bb_ca` 和 `local_latents`，并通过 flow matching 学习从噪声到真实 binder 的向量场。长期 `bb_ca_flow=0` 只能做 geometry-preserving repair，不能实现只给 target 多构象就生成 binder pose/backbone/sequence 的目标。

AE decoder 仍是原始单 binder AE。它可以从 `z_latent + ca_coors_nm` 解码序列和全原子结构，但不会天然保证 K 个 state 解码为同一条序列。因此 Stage12 增加显式 shared sequence token，并把该信号反馈到 K 个 state-specific flow head，避免最后一层投票式共享。

## 验证要求

本阶段最低验证包括：

- `py_compile` 覆盖 Stage12 新脚本和改动模块。
- `stage12_flow_coupling_probes.py` 通过 shape、no-leak、state permutation、de novo corruption probe。
- `stage12_de_novo_multistate_training.py --dry-run`、`stage12_de_novo_multistate_sampling.py --dry-run`、`stage12_exact_benchmark.py --dry-run` 通过。
- 若 probe 或脚本日志出现 warning/error，必须判断是否影响 target-only de novo、多状态共享序列、state-specific complex geometry 这三个核心目标。

## 当前限制

本次代码把 Stage12 的核心耦合结构和验证入口接入完成，但完整 161/36 pilot 训练和 48 exact B0/B1/B2 benchmark 需要后续 GPU 作业执行。正式比较旧 B1 或 B0 之前，必须先确认 Stage12 B1 的 sampled latent identity 和 exact holdout sequence identity 不再复现 Stage11G 的 loss 下降但 identity 下降问题。


## 本轮实测结果（2026-05-07）

已完成的非 GPU 验证：

- `py_compile` 通过：
  - `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
  - `src/proteinfoundation/flow_matching/multistate_loss.py`
  - `scripts/strategy01/stage12_flow_coupling_probes.py`
  - `scripts/strategy01/stage12_de_novo_multistate_training.py`
  - `scripts/strategy01/stage12_de_novo_multistate_sampling.py`
  - `scripts/strategy01/stage12_exact_benchmark.py`
- `stage12_flow_coupling_probes.py` 通过，核心实测 shape：
  - `bb_ca_states = [2, 3, 32, 3]`
  - `local_latents_states = [2, 3, 32, 8]`
  - `state_seq_logits = [2, 3, 32, 20]`
  - `seq_logits_shared = [2, 32, 20]`
  - `shared_seq_tokens = [2, 32, 768]`
- state 顺序置换测试通过：`seq_logits_shared` 最大绝对差约 `8.34e-7`，说明 shared sequence 路径对 state 顺序近似不敏感。
- no-leak guard 通过：`de_novo_multistate_mode=True` 时传入 `init_bb_ca_states` 会报错。
- de novo corruption probe 通过：`stage12_primary_state_tensors=True`，并生成 state-wise `x_t_states`。
- 三个入口 dry-run 通过：
  - `stage12_de_novo_multistate_training.py --dry-run`
  - `stage12_de_novo_multistate_sampling.py --dry-run`
  - `stage12_exact_benchmark.py --dry-run`

## 报错与修复记录

1. 远程补丁脚本第一次执行失败：
   - 表现：`SyntaxError: invalid syntax`。
   - 根因：补丁脚本中要写入远程文件的 docstring 嵌套了三引号。
   - 修复：把该 docstring 改为普通注释后重跑，补丁成功应用。

2. `stage12_flow_coupling_probes.py` 初次运行失败：
   - 表现：`ModuleNotFoundError: No module named 'proteinfoundation.flow_matching.multistate_loss'`。
   - 根因：脚本在插入 `REPO/src` 到 `sys.path` 之前已经 import 了 `proteinfoundation`。
   - 修复：把 `REPO`、`REPO/src` 的 `sys.path` 插入移动到 repo import 之前。

3. dry-run 脚本初次运行失败：
   - 表现：`NameError: name 'REPO' is not defined`。
   - 根因：自动补 path 脚本只匹配了含 `torch` import 的 probe 脚本，未覆盖 contract-only 脚本。
   - 修复：给 training/sampling/benchmark 三个脚本显式补 `REPO` 和 `sys.path`。

4. 多次 SSH 命令失败：
   - 表现：`Connection timed out`、`server not responding`，以及 PowerShell 对远程 `|`、引号、`python -c` 的破坏。
   - 根因：远程网络/登录节点响应慢叠加 PowerShell 复杂命令转义问题。
   - 修复：继续采用“本地临时脚本 -> scp -> 远程 Python 执行”的方式；复杂命令不再直接嵌套在 PowerShell SSH 字符串中。

## Warning 影响判断

probe 日志中出现以下 warning：

- `CCD_MIRROR_PATH/PDB_MIRROR_PATH not set`：本次 CPU shape/no-leak/probe 不调用依赖 CCD/PDB mirror 的结构数据库功能，不影响本阶段结论；后续如做真实 PDB/mmCIF 解析或 ligand/CCD 相关流程，需要重新检查。
- `torch.load weights_only=False FutureWarning`：当前 checkpoint 是本策略仓/基线仓的可信本地 checkpoint，不影响结果；后续可在只读取 state_dict 的 probe 中改成更安全的加载方式。
- PyTorch nested tensor warning：性能提示，不影响数值正确性。
- optional CA/residue feature 返回 zeros：本阶段为了 no-leak 默认禁用真实 binder CA/residue type 特征，因此该 warning 不 invalid 当前 target-only probe；但它会限制绝对生成质量，后续应设计非泄漏的 predicted/self-conditioned CA/sequence feature，而不能直接塞真实 binder feature。

## 下一步必须做的 GPU 验证

当前 Stage12 完成的是架构接线和 CPU 级正确性验证，还没有证明模型质量提升。下一步应提交单 A100 作业：

1. 用现有 `161 train / 36 val` 跑 Stage12 1-sample 和 4-sample overfit。
2. 检查 sampled latent decode identity 是否从 Stage11G 的 `0.4414` 上升，而不是复现 `0.2183` 的退化。
3. 若 overfit 通过，再跑 48 exact diagnostic 的 B1 Stage12 de novo sampling。
4. B0/B1/B2 正式比较必须使用 `bb_ca_states[k]` 与 state-specific complex 输出，不能使用 legacy weighted-average `bb_ca/local_latents`。
