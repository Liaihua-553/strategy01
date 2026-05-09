# Strategy01 Stage15：Self-conditioned flow 训练-采样分布错配修复报告

更新时间：2026-05-10

## 阶段目标

本阶段继续围绕唯一核心科学目标：只输入多状态 target 构象，让模型从噪声生成 binder 的空间位置、backbone、local latent，并输出一条兼容多个 target states 的 shared binder sequence。

Stage14 证明当前 best checkpoint 在 teacher-forced / validation 条件下 sequence 指标较好，但 target-only de novo rollout 仍低：K=1 val n16 self-conditioning audit 的 final sampled identity 约 `0.2273`。因此本阶段不扩大数据规模，而是先修训练条件和采样条件不一致的问题。

## 已完成诊断

### 1. VF deterministic sampling 分支

新增配置：`configs/strategy01/stage14_model_sampling_vf.yaml`。该配置只做两处改变：

- `sampling_mode: sc` 改为 `sampling_mode: vf`
- `sc_scale_noise: 0.1` 改为 `0.0`

审计结果：

| 配置 | nsteps | final sampled AE/shared identity | 结论 |
|---|---:|---:|---|
| SC + self-conditioning | 16 | 0.2273 | 当前最好 |
| SC + self-conditioning | 32 | 0.1409 | 步数增加会漂移 |
| VF deterministic | 16 | 0.0682 | 明显更差 |
| VF deterministic | 32 | 0.0682 | 明显更差 |

判断：主要问题不是 stochastic corrector noise；原始 `sc` corrector 对当前模型反而有帮助。继续沿 VF 分支优化不合理。

### 2. 发现训练-采样错配

原始 Complexa sampling 使用 self-conditioning：下一步会把上一步模型预测的 clean state 作为 `x_sc` 输入。Stage14 已修复 Stage12/13 自定义 rollout 的 self-conditioning bug。

但训练代码的 teacher-forced 分支此前没有让模型看到 `x_sc_states=模型自己预测的 clean state`。这导致：

- 训练：模型在真实 interpolant `x_t` 上、通常无 self-conditioning 输入。
- 采样：模型在自己 rollout 的 `x_t` 上，并且每一步都有模型自产生的 `x_sc_states`。

这会造成分布错配。hard replay 以前失败，说明不能直接把远离流形的 rollout state 强拉到真值；更安全的改法是在 teacher-forced interpolant 上加入 model-generated self-conditioning。

## 代码改动

文件：`scripts/strategy01/stage12_de_novo_multistate_training.py`

新增函数：`attach_teacher_forced_self_conditioning()`

功能：

1. 在真实 teacher-forced `x_t_states/t_states` 上先用当前模型无梯度预测 clean sample。
2. 通过各 data mode 的 base flow matcher 转成 clean `x_1` 预测。
3. 将该预测作为 `x_sc_states` 回填到同一个 batch。
4. 再进行正常 supervised loss 反传。

科学意义：该过程不输入真实 binder pose、source complex、真实 residue optional feature，也不把真实 binder sequence 作为模型输入；`x_sc_states` 完全由模型当前预测生成。因此它保持 target-only de novo 契约，同时让训练覆盖采样实际使用的 self-conditioning 输入。

新增 CLI：

- `--enable-teacher-self-conditioning`
- `--teacher-self-cond-prob`

CPU probe：

- run name：`stage15_teacher_selfcond_cpu_probe`
- 状态：passed
- 报告：`reports/strategy01/probes/stage15_teacher_selfcond_cpu_probe_results.json`
- 诊断：`teacher_self_conditioned=true`，并记录 `teacher_self_cond_bb_ca_mean_abs`、`teacher_self_cond_local_latents_mean_abs`。

## Warning / Error 影响判断

CPU probe 和 VF audit 中仍有 optional CA/residue feature disabled warning。该 warning 是 de novo no-leak 预期行为：真实 binder CA 坐标和 residue type 不作为模型输入。它不影响科学结果。

简单日志 grep 曾把 `Nanometers` 中的 `nan` 误判为 NaN；实际没有数值 NaN、Traceback、OOM 或 failed。

## 当前运行状态

GPU 训练任务已提交：`2019530`。

任务配置：

- 初始 checkpoint：`ckpts/stage12_de_novo_multistate/runs/stage14_native_latent_warmup_161/stage12_pilot_final.pt`
- `--stage13-native-state-path`
- `--stage13-heads-only`
- `--stage14-native-latent-warmup`
- `--enable-teacher-self-conditioning`
- `--lambda-ae-seq 3.0`
- `--lambda-seq-ae-consistency 0.50`
- `161 train / 36 val`
- batch size 4
- pilot 800 steps

当前 SLURM 状态：combined job `2025298` 正在等待 A100 resources。该 job 会在同一次 GPU allocation 中完成：

1. `stage15_teacher_selfcond_from_warmup` 训练。
2. n16 de novo audit：`stage15_single_state_audit_teacher_selfcond_val4_n16.json`。
3. n32 de novo audit：`stage15_single_state_audit_teacher_selfcond_val4_n32.json`。

## 下一步验收

训练完成后必须执行：

1. 解析 `stage15_teacher_selfcond_from_warmup_results.json`，看 validation identity、AE identity、loss 是否稳定。
2. 用新 checkpoint 跑 K=1 de novo audit：n16 和必要时 n32。
3. 对比当前 best：`0.2273`。
4. 若改善，进入 K>1 / 48 exact smoke；若不改善，说明 self-conditioning 不是主瓶颈，下一步转向低/中 t sequence supervision 或 latent manifold regularization。

## 当前结论

Stage15 已修复一个真实训练-采样契约缺口，但还不能声明达到科学目标。是否有效必须以 GPU 训练后的 de novo rollout 指标为准。

## 本阶段错误与修复日志

### 错误 1：`OSError: [Errno 28] No space left on device`

触发 job：`2019530`。

现象：训练尚未进入模型主体，在 import `lightning -> torchmetrics -> torchvision -> torch._dynamo -> torch.distributed.nn.jit.instantiator` 时失败。

根因：`torch.distributed.nn.jit.instantiator` 会在 import 阶段向临时目录写 generated module。登录节点 `/tmp` 和 `/data` 空间充足，但 gu02 compute node 的本地 `/tmp` 可能满或不可写。

修复：在 `scripts/strategy01/slurm/stage15_teacher_selfcond.sbatch` 中显式设置：

- `TMPDIR=/data/kfliao/general_model/strategy_repos/.runtime/strategy01/tmp`
- `TMP/TEMP=$TMPDIR`
- `XDG_CACHE_HOME=/data/kfliao/general_model/strategy_repos/.runtime/strategy01/xdg_cache`
- `TORCH_HOME=/data/kfliao/general_model/strategy_repos/.runtime/strategy01/torch_home`
- `PYTORCH_KERNEL_CACHE_PATH=/data/kfliao/general_model/strategy_repos/.runtime/strategy01/torch_kernel_cache`

影响判断：该错误发生在 import 阶段，未产生有效训练结果，不影响已有 Stage14 科学结论。

### 错误 2：job 长时间 pending `Resources`

触发 job：`2022466`。

现象：`gu02` 处于 `MIXED`，但 job 不启动。

根因：`gu02` 当时 CPU 分配 `34/36`，原 job 请求 `--cpus-per-task=4`，剩余 CPU 不足；之后 nodeInfo 显示 `gu02 8/8 GPU` 被占用，说明 A100 本身也可能满载。

修复：将 Stage15 Slurm 脚本改为 `--cpus-per-task=1`，避免因 CPU 请求过高阻塞单 GPU 训练。当前只保留一个 pending job `2025298`，不并发排队。

### 错误 3：combined script patch marker 未匹配

现象：第一次尝试把 audit 追加到 Slurm 脚本时，patch marker 未匹配，短暂留下两个 pending job。

修复：立即 `scancel 2024014 2024987`，随后重写整个 `stage15_teacher_selfcond.sbatch`，确认只保留一个 job `2025298`。

影响判断：这是执行脚本管理错误，没有运行训练，也没有影响模型结果。
