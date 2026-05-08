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

## Stage12B 执行记录：warning 清理、de novo 契约、state-wise flow 接入与 GPU 验证

### 1. warning 与 no-leak 处理结论

- `no-leak guard` 不是 bug，而是 Stage12 target-only de novo 主线的科学契约。`de_novo_multistate_mode=True` 时继续禁止 `init_bb_ca_states/source_*`，否则任务会退化为 source-pose repair 或 inverse folding。
- probe 字段已改成 `expected_no_leak_guard_passed`，避免把预期报错误读为失败。
- repair/refinement baseline 没被破坏：`de_novo_multistate_mode=False` 时 `init_bb_ca_states` 正常通过，probe 字段 `repair_mode_allows_init_pose=true`。
- `torch nested tensor` warning 已通过 `TransformerEncoder(..., enable_nested_tensor=False)` 消除。该修复只清理 PyTorch 性能 warning，不改变 `norm_first=True` 的数学结构。
- 剩余 warning 分类：`CCD_MIRROR_PATH/PDB_MIRROR_PATH` 未设置只影响需要本地 PDB/CCD mirror 的工具，不影响本阶段 tensor 训练；optional CA/residue feature 返回零是 de novo no-leak 设定下的预期行为；未出现 `Traceback/OOM/NaN/nested tensor`。

### 2. 发现并修正的科学实现问题

代码审查发现，Stage12 原实现虽然在 loss 中保存了 `x_t_states`，但 `LocalLatentsTransformerMultistate` 的 state tokens 主要来自共享 binder token + state bias，没有显式读取每个 state 的当前 noisy `bb_ca/local_latents`。这会让所谓 K 条 state-specific trajectories 实际上退化为同一 binder token 的 K 个 state-biased readout，不完全符合“多条复合物结构 flow 共同优化一个共享 binder 序列”的科学目标。

已修正：新增 `state_xt_bb_ca_projector` 与 `state_xt_latent_projector`，把 `x_t_states['bb_ca']` 和 `x_t_states['local_latents']` 投影进 state tokens/state sequence tokens。它们是训练/采样中的 noisy flow variables，不是 source pose 或真实标签，因此不破坏 target-only de novo 契约。

同时在采样器中新增显式开关：`--de-novo-multistate-mode --pass-x-t-states`。默认不改变 Stage09/10 repair baseline；Stage12 采样时才把 state-wise flow variables 传入模型。

### 3. CPU probe 与 GPU 训练结果

CPU 级验证：`py_compile`、shape probe、state permutation、no-leak negative test、repair-mode positive test、state-wise corruption、CPU train smoke 均通过。`nested tensor` warning 未再出现。

接入 `x_t_states` 前的 GPU 结果：

- `overfit1`: steps=80, batch=1, drop=0.4993, final_total=13.6114, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.9333333373069763}`, mem=7.388GB, step_time=0.407s
- `overfit4`: steps=120, batch=4, drop=0.6229, final_total=9.2269, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.7833333015441895}`, mem=7.395GB, step_time=0.402s
- `pilot`: steps=200, batch=4, drop=0.3007, final_total=17.7935, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.7916666865348816}`, mem=7.395GB, step_time=0.269s
- final validation: total=36.6954, identity=`{'shared_identity_mean': 0.5568181872367859, 'ae_state_identity_mean': 0.5}`

接入 `x_t_states` 后的 GPU 结果：

- `overfit1`: steps=80, batch=1, drop=0.7203, final_total=8.1112, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.9000000357627869}`, mem=7.409GB, step_time=0.448s
- `overfit4`: steps=120, batch=4, drop=0.8522, final_total=4.0744, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.7749999761581421}`, mem=7.416GB, step_time=0.395s
- `pilot`: steps=200, batch=4, drop=0.3359, final_total=18.8602, final_identity=`{'shared_identity_mean': 1.0, 'ae_state_identity_mean': 0.7833333015441895}`, mem=7.416GB, step_time=0.280s
- final validation: total=36.8909, identity=`{'shared_identity_mean': 0.6136363744735718, 'ae_state_identity_mean': 0.4848484992980957}`

关键变化：`x_t_states` 接入后，final validation 的 `shared_identity_mean` 从 `0.5568` 提升到 `0.6136`，首次超过 Stage12 机制阈值 `0.60`。但 `ae_state_identity_mean` 仍只有 `0.4848`，说明 shared head 已改善，state-specific local latent 到 AE decoder 的序列一致性仍不足。

### 4. Stage12E de novo sampling smoke 结果

已跑 8 个 V_exact sample 的 target-only de novo sampling smoke。该流程不使用 source pose/init，不启用 oracle guidance，只输入 target ensemble，并打开 `--de-novo-multistate-mode --pass-x-t-states`。

- sampling sample_count=8, elapsed_sec=18.84, mean_sec_per_sample=2.355
- mean sequence identity to reference=0.0605
- exact benchmark B1 state_count_ok=24
- B1 contact-F1 mean=0.1118, worst=0.0000, best=0.3317
- B1 direct interface RMSD mean=25.0421 A
- B1 contact persistence mean=0.1168
- B1 clash_rate=0.6250

结论：Stage12B 的训练链路和 state-wise flow 接线已经更正确，但自由 de novo rollout 仍未达到科学目标。训练/验证 corruption 条件下 sequence identity 能提升，不代表从纯噪声采样时 sequence/geometry 已进入正确 manifold。当前主要瓶颈是 **training-sampling distribution mismatch**：模型在 teacher-forced interpolant 上能学，但 free rollout 的 sampled local latent / bb_ca state 仍偏离真实多状态复合物 manifold。

### 5. 报错与修复记录

- `1948723` 长时间 pending：原因是 `gu02` 只剩 6 个 idle CPU，而脚本申请 8 CPU；修复为 `cpus-per-task=4` 后 `1948724` 正常运行。
- `1948726` sampling smoke 报 `ModuleNotFoundError: No module named torch` 和 `Permission denied`：原因是 sbatch 文件生成时 `$PY` 被提前展开为空，导致脚本直接用系统 Python 执行 `.py` 文件；修复为本地生成 sbatch 后 scp，并保留 `$PY` 变量。
- 第二次提交 sbatch 报 `Batch script contains DOS line breaks`：原因是 Windows CRLF；修复为远程 `perl -pi -e 's/\r$//'` 转 LF 后提交，`1948727` 正常完成。
- 报告追加曾出现中文乱码：原因是本地临时脚本用 ASCII 写入中文；修复为 UTF-8 Markdown 片段重写本节。

### 6. 下一步必须解决的问题

Stage12C 不应继续单纯加训练步数。下一步应实现 sampled-condition replay / rollout replay：用模型自身从 pure-noise rollout 产生的中间 `x_t_states` 作为训练条件，再对同一 shared sequence、state-specific geometry、AE-decoded logits 反传。目标是让训练分布覆盖真实采样分布，避免 Stage12B 出现“corruption 验证 identity 上升，但 de novo sampling identity 只有 0.0605”的错配。

建议优先级：

1. 在训练中加入 short-rollout replay batch：每个 batch 混合 teacher-forced corruption 与 4-8 step self-generated states。
2. 对 replay states 加强 `L_ae_seq_state`、state-shared KL、interface/contact/clash loss，但保持 clash 在最终选择中 hard fail。
3. 采样时继续使用 `x_t_states`，并记录 per-step shared identity proxy、flow gate、state disagreement，定位 sequence 是在哪个 denoising 阶段崩坏。
4. 只有当 8-sample de novo smoke 的 sequence identity 和 clash/contact 同时改善后，再跑 48 exact benchmark。


## Stage12C：short-rollout / sampled-condition replay 闭环结果（更新于 2026-05-08 19:20:52）

### 本阶段要解决的问题
Stage12B 已经完成了 coupled full multistate product-space flow 的架构接线，但自由 de novo sampling 仍然失败：模型训练主要覆盖 teacher-forced interpolant，未覆盖自己 rollout 后产生的 off-manifold `x_t_states`。因此 Stage12C 增加 `short-rollout replay`，目标是让训练直接看到模型自产生的中间状态，再用同一真实 `x_1_states`、shared binder sequence 和 interface labels 反传。

### 代码改动
- `scripts/strategy01/stage12_de_novo_multistate_training.py`
  - 新增 `build_replay_condition_batch()`，从 target-only noise 出发调用当前模型 rollout，收集中间 `x_t_states/t_states` 作为 replay 训练条件。
  - 新增 `--enable-replay`、`--replay-mix-prob`、`--replay-warmup-steps`、`--replay-rollout-steps`、`--replay-collect-fraction`、`--replay-max-loss-t`、`--replay-teacher-blend` 与 replay extra loss 权重。
  - 修正 v1 中 replay 调度公式，避免前 200 step 几乎没有 replay 训练。
  - v4 新增 `teacher_blend`，把 rollout 中间状态向 clean target 轻度混合，使 replay 从完全 off-manifold 转为 near-manifold curriculum。
  - validation 额外输出 `validation.replay`，不再只看 teacher-forced validation。
- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
  - 保留 no-leak guard，并新增 optional real CA / residue-type feature 禁止逻辑，确保 `de_novo_multistate_mode=True` 下不输入真实 binder 坐标或真实 binder residue type。
- `scripts/strategy01/stage12_flow_coupling_probes.py`
  - 增加 optional feature no-leak negative tests；预期报错被记录为 guard passed。
- `scripts/strategy01/stage12_exact_benchmark.py`
  - 增加 legacy average misuse guard。主评估若试图读取 weighted-average `bb_ca/local_latents` 会直接失败，必须使用 `bb_ca_states[k]/local_latents_states[k]`。
- `scripts/strategy01/stage12c_de_novo_smoke.py`
  - 新增真实 target-only de novo rollout smoke：从噪声完整 rollout，最终只用 state-specific outputs 评估 shared sequence identity、AE-state identity 和 multistate loss。

### Warning / error 处理
- `torch nested tensor` warning 在 Stage12B 已通过显式禁用 nested tensor 消除，不影响 Stage12C。
- `use_ca_coors_nm_feature disabled` / `use_residue_type_feature disabled` warning 在 Stage12C 仍出现。这个 warning 对本阶段不是错误，而是 no-leak 契约的预期现象：真实 binder CA 和真实 residue type 不能作为 de novo 输入。它可能限制绝对性能，但不导致结果无效。
- 运行脚本曾因 `stage12_flow_coupling_probes.py` 参数误用失败一次：该脚本没有 `--device/--run-name` 参数。根因是把训练脚本 CLI 误套到 probe 脚本；已按真实 `--report-json` 参数重跑并通过。
- Slurm 作业 `1956470` 申请 8 CPU 导致在 `gu02` 上因 CPU 不足 pending；已取消并改为 2 CPU + 1 A100 的 `1957169` 成功运行。根因是 `gu02` 当时有 A100 空闲但 CPU 剩余不足。

### CPU / contract gates
- `py_compile` 覆盖 Stage12 training、flow probe、exact benchmark、multistate transformer。
- no-leak guard 通过：`init_bb_ca_states/source_*` 和 optional real CA / residue-type feature 在 `de_novo_multistate_mode=True` 下会触发预期失败。
- repair mode 未删除；`de_novo_multistate_mode=False` 仍可作为 pose repair baseline。
- benchmark legacy-average negative test 通过：`prediction_source=legacy_average` 会报错，避免把平均结构误作科学主输出。

### GPU 训练结果
| run | 关键设置 | overfit1 drop | overfit4 drop | pilot drop | validation shared identity | validation AE-state identity | validation replay shared identity | validation replay AE-state identity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 | replay 0.20, warmup 60, clamp 0.75 | 0.790 | 0.887 | 0.339 | 0.330 | 0.511 | 未记录 | 未记录 |
| v4 | replay 0.10, warmup 100, teacher_blend 0.35 | 0.794 | 0.893 | 0.361 | 0.386 | 0.511 | 0.159 | 0.341 |

解释：v4 比 v3 的 teacher-forced validation shared identity 从约 0.330 提升到 0.386，说明 near-manifold replay curriculum 有正作用。但 validation replay shared identity 只有 0.159，说明模型仍未学会对 unseen target 的自产生 off-manifold 中间状态做可靠恢复。

### 真实 de novo smoke
| checkpoint | nsteps | val samples | shared identity | AE-state identity | contact loss | distance loss | clash loss | AE seq loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage12C-v4 | 16 | 8 | 0.176 | 0.098 | 0.504 | 0.698 | 0.010 | 14.368 |
| Stage12C-v4 | 64 | 8 | 0.091 | 0.049 | 0.477 | 0.667 | 0.010 | 16.927 |

Stage12B 的 8-sample de novo shared identity 约 0.0605。Stage12C-v4 的 16-step smoke 提升到 0.176，是实质进步，但没有达到计划 gate `>=0.20`。64-step 反而下降到 0.091，说明失败不是采样步数不足，而是完整 rollout 过程中 `local_latents_states` 仍逐步离开 AE/sequence-compatible manifold。

### 是否达到科学目标
没有。Stage12C 证明了 short-rollout replay 是正确方向的一部分，但目前只做到“训练子集和近邻 replay 可学”，还没有做到“target-only de novo 输入 K 个 target states 后稳定生成一个共享 binder 序列和 K 个合理复合物”。因此不能进入正式 B0/B1/B2 胜负比较，也不能扩大数据规模宣称成功。

### 下一步修正方向
1. 从 on-the-fly replay 切到固定 replay-cache curriculum：先生成 easy/medium/hard 三档 replay cache，使 validation replay 分布固定可比，而不是每次随当前模型漂移。
2. 对 `local_latents_states` 增加 AE-manifold projection / latent norm regularization / sequence-compatible latent repair，目标是降低 de novo smoke 的 `multistate_ae_seq_justlog`。
3. 将 shared sequence feedback 更早接入 denoising step，而不只在 readout 后约束；当前 `state_seq_disagreement` 很低但 shared entropy 仍高，说明 state branches 没有真正给 shared sequence 提供强判别信息。
4. 在进入 48 exact benchmark 前，必须先让 8-sample de novo smoke 过 gate：shared identity `>=0.20`，且 contact/clash 不退化。

## Stage12D：固定 replay-cache、latent-sequence repair 与 local-latent stop-t 诊断（更新于 2026-05-08）

### 本阶段要解决的问题
Stage12C 的关键失败不是 AE decoder 失效，而是自由 rollout 后 `local_latents_states` 偏离冻结 AE 的 sequence-compatible latent manifold。Stage12D 围绕这个问题做三个最小闭环：

1. 固定 replay-cache curriculum：把 replay 条件固定成 easy/medium/hard 三档，避免在线 replay 每步漂移。
2. latent-sequence repair：让 shared sequence token 直接修正 `local_latents_states` readout。
3. local-latent stop-t 诊断：验证采样后段 `local_latents` 继续推到训练未覆盖的高 `t` 区域时，是否导致序列流形损坏。

### 代码改动
- `scripts/strategy01/stage12_de_novo_multistate_training.py`
  - 新增 `build_fixed_replay_cache()`，支持 `--enable-fixed-replay-cache`、`--replay-cache-batches`、`--replay-cache-levels`、`--replay-curriculum-switches`。
  - `forward_loss_stage12()` 支持 `replay_cache_entry`，训练可以直接使用固定 replay batch。
  - `run_loop()` 按训练进度从 easy/medium/hard cache 中取 replay 条件，并记录 `fixed_cache_level`、`fixed_cache_teacher_blend`。
- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
  - 新增 `latent_sequence_repair_projector` 和 `latent_sequence_repair_gate`。
  - `local_latents_states = local_latents_base + 0.25 * gate * repair`，使 shared sequence 信号直接作用到 AE decoder 所读的 latent。
  - 新增 early shared sequence feedback，让 sequence signal 在第一次 state sequence readout 前就影响 state tokens。
- `scripts/strategy01/stage12c_de_novo_smoke.py`
  - 每个关键 timestep 记录 `shared_identity_mean`、`ae_state_identity_mean`、feedback norm、latent repair gate。
  - 新增 `--local-latents-stop-t` 诊断参数，测试 `local_latents` 末端过冲。
- `scripts/strategy01/probe_multistate_arch.py`
  - 显式 `torch.load(..., weights_only=False)`，消除 PyTorch FutureWarning。

### Warning / error 处理
- `torch.load weights_only` FutureWarning：已修复。
- `stage12c_de_novo_smoke.py` 首次提交误用 `--init-ckpt/--run-name`：这是 CLI 错误，不是模型错误；已改为 `--checkpoint/--report-json` 并重跑成功。
- `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` 未设置：当前使用已 tensorized 数据，不调用 PDB/CCD mirror，不影响 Stage12D 结果。
- optional CA/residue feature disabled warnings：这是 no-leak 预期行为，不影响结果有效性。

### 训练与 smoke 结果
| 实验 | 关键设置 | shared identity | AE-state identity | contact loss | distance loss | clash loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage12C-v4 16-step | online replay + blend 0.35 | 0.176 | 0.098 | 0.504 | 0.698 | 0.010 | 旧最好 de novo smoke |
| Stage12D fixed-cache 16-step | hard=0.10 | 0.165 | 0.100 | 0.621 | 0.756 | 0.013 | hard replay 过早，未改善 |
| Stage12D fixed-cache 64-step | hard=0.10 | 0.125 | 0.112 | 0.537 | 0.628 | 0.013 | 长 rollout 仍退化 |
| Stage12D soft-cache 16-step | hard=0.35 | 0.148 | 0.087 | 0.570 | 0.754 | 0.012 | 软 cache 未改善 |
| Stage12D stop-t 0.75, 16-step | local_latents stop at 0.75 | 0.193 | 0.100 | 0.622 | 0.756 | 0.013 | 接近 sequence gate |
| Stage12D stop-t 0.65, 16-step | local_latents stop at 0.65 | 0.227 | 0.087 | 0.623 | 0.755 | 0.013 | shared readout 过 0.20，但 AE latent 仍差 |

### 关键诊断
`local_latents_stop_t=0.65` 首次把 8-sample shared identity 提到 `0.227`，超过 Stage12C 计划中的 `0.20` sequence gate。但 AE-state identity 仍只有 `0.087`，contact/distance 没同步改善。这说明 stop-t 主要救了 shared readout，并没有让每个 state-specific `local_latents_states` 真正回到 AE 可解码的共享序列流形。

因此 Stage12D 仍未达到科学目标。当前模型还不是可靠的 target-only 多状态 binder 生成器；它只证明了 `local_latents` 末端过冲是一个真实瓶颈。

### 下一步
Stage12E 应直接做 AE latent manifold 修复，而不是继续扩大训练或盲目调 replay 权重：

1. 加入 clean latent global statistics / norm / drift regularization，限制 sampled `local_latents_states` 远离训练分布。
2. 把 `latent_sequence_repair_projector` 升级为 decoder-consistent projection head，以 AE-decoded CE 和 state-shared KL 为主监督。
3. 将 `local_latents_stop_t` 从诊断参数变成训练-采样一致的 bounded latent schedule，再逐步放宽终点。
4. 进入 48 exact benchmark 前，必须同时看到 shared identity、AE-state identity、contact/distance 三者改善。
