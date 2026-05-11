# Strategy01 Stage20 selector 与 exact/label geometry 分析报告

## 阶段目标

Stage18 已经把 severe clash 从主要失败模式降下来，但这还不等于达到科学目标。真正目标要求同一个共享 binder 序列在多个 target state 下形成合理、正确的界面。Stage20 因此把 Stage19 生成的 state-specific PDB 接入几何评估，分析当前 B1 是否只是“贴近 target”，还是贴到了正确界面。

## 新增代码

- `scripts/strategy01/stage19_candidate_geometry_benchmark.py`
  - 输入 Stage19 candidate JSON 和当前 tensor dataset。
  - 读取 selected candidate 的 `stateXX_binder_ca.pdb`。
  - 使用 `x_target_states`、`x_1_states['bb_ca']`、`interface_contact_labels` 计算：contact-F1、direct/aligned binder CA RMSD、severe clash、generated contact persistence。
  - 明确报告：当前评估对象是 Stage10/Stage07 tensor labels，不冒充为 V_exact-only。
- `scripts/strategy01/stage20_selection_sweep_from_pdbs.py`
  - 对所有 generated candidate PDB 做 posthoc selection sweep。
  - selection score 只用 label-free 几何和模型置信度，label contact/RMSD 只用于诊断比较。
- `scripts/strategy01/stage12c_de_novo_smoke.py`
  - `target_binder_geometry_proxy()` 增加 `target_contact_persistence_mean`。
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`
  - 默认 selector 改为 hard-gate + contact-shell/persistence score。

## Stage19 几何评估结果

输入：`reports/strategy01/probes/stage19_safe_select_pdb_val36_iter8.json`

aggregate：

```json
{
  "mean_contact_f1": {
    "n": 36,
    "mean": 0.03437588911770354,
    "best": 0.15064997592681753,
    "worst": 0.0
  },
  "worst_contact_f1": {
    "n": 36,
    "mean": 0.01222542505358653,
    "best": 0.0967741935483871,
    "worst": 0.0
  },
  "mean_direct_rmsd_A": {
    "n": 36,
    "mean": 34.03438602332715,
    "best": 17.242313027381897,
    "worst": 87.14557528495789
  },
  "worst_direct_rmsd_A": {
    "n": 36,
    "mean": 39.413562052779724,
    "best": 18.4932804107666,
    "worst": 134.3956184387207
  },
  "mean_aligned_rmsd_A": {
    "n": 36,
    "mean": 19.561123343001466,
    "best": 7.689422973149535,
    "worst": 60.77444728276326
  },
  "severe_clash_rate": {
    "n": 36,
    "mean": 0.009259259259259259,
    "best": 0.0,
    "worst": 0.3333333333333333
  },
  "contact_persistence": {
    "n": 36,
    "mean": 0.22207094793593002,
    "best": 0.6296296296296297,
    "worst": 0.0
  },
  "mean_generated_contact_count": {
    "n": 36,
    "mean": 62.67592592592593,
    "best": 131.33333333333334,
    "worst": 5.5
  }
}
```

关键解释：

- severe clash rate 已经很低，mean `0.0093`，说明 Stage18 safe sampling 确实修掉了大部分硬碰撞。
- 但 mean contact-F1 只有 `0.0344`，worst-state contact-F1 mean 只有 `0.0122`。
- direct RMSD mean 约 `34 Å`，说明生成 binder 常常不在标签界面附近。
- generated contact count mean 约 `62.7`，说明模型并非完全远离 target，而是经常贴到了错误 target 区域。

这说明当前 B1 还没有学会 target-relative interface placement。科学上不能把它判定为成功。

## Stage20 selector sweep

```json
{
  "current_hard_gate": {
    "selected_count": 36,
    "safe_rate": 0.9722222222222222,
    "mean_contact_f1": {
      "n": 36,
      "mean": 0.03437588911770354,
      "best": 0.15064997592681753,
      "worst": 0.0
    },
    "worst_contact_f1": {
      "n": 36,
      "mean": 0.01222542505358653,
      "best": 0.0967741935483871,
      "worst": 0.0
    },
    "mean_direct_rmsd_A": {
      "n": 36,
      "mean": 34.03438602332715,
      "best": 17.242313027381897,
      "worst": 87.14557528495789
    },
    "contact_persistence": {
      "n": 36,
      "mean": 0.22207094793593002,
      "best": 0.6296296296296297,
      "worst": 0.0
    },
    "generated_contact_count_mean": {
      "n": 36,
      "mean": 62.67592592592593,
      "best": 131.33333333333334,
      "worst": 5.5
    },
    "shared_identity_posthoc": {
      "n": 36,
      "mean": 0.21529471623297367,
      "best": 0.4444444477558136,
      "worst": 0.0
    }
  },
  "persistence_contact": {
    "selected_count": 36,
    "safe_rate": 0.9722222222222222,
    "mean_contact_f1": {
      "n": 36,
      "mean": 0.033372625088469506,
      "best": 0.1390728476821192,
      "worst": 0.0
    },
    "worst_contact_f1": {
      "n": 36,
      "mean": 0.01691312714605764,
      "best": 0.10596026490066225,
      "worst": 0.0
    },
    "mean_direct_rmsd_A": {
      "n": 36,
      "mean": 34.11583842502701,
      "best": 18.86910875638326,
      "worst": 90.64441204071045
    },
    "contact_persistence": {
      "n": 36,
      "mean": 0.31817577503066397,
      "best": 0.7974683544303798,
      "worst": 0.0
    },
    "generated_contact_count_mean": {
      "n": 36,
      "mean": 69.52777777777777,
      "best": 197.33333333333334,
      "worst": 5.5
    },
    "shared_identity_posthoc": {
      "n": 36,
      "mean": 0.20898158473169637,
      "best": 0.4444444477558136,
      "worst": 0.0
    }
  },
  "contact_shell": {
    "selected_count": 36,
    "safe_rate": 0.9722222222222222,
    "mean_contact_f1": {
      "n": 36,
      "mean": 0.04363543163242616,
      "best": 0.17587034813925567,
      "worst": 0.0
    },
    "worst_contact_f1": {
      "n": 36,
      "mean": 0.023093534777922406,
      "best": 0.1487603305785124,
      "worst": 0.0
    },
    "mean_direct_rmsd_A": {
      "n": 36,
      "mean": 36.070443557368385,
      "best": 15.546963810920715,
      "worst": 87.14557528495789
    },
    "contact_persistence": {
      "n": 36,
      "mean": 0.32248059688587666,
      "best": 0.7974683544303798,
      "worst": 0.0
    },
    "generated_contact_count_mean": {
      "n": 36,
      "mean": 60.55555555555556,
      "best": 114.66666666666667,
      "worst": 5.5
    },
    "shared_identity_posthoc": {
      "n": 36,
      "mean": 0.2115068373322073,
      "best": 0.4444444477558136,
      "worst": 0.0
    }
  }
}
```

结论：

- `contact_shell` selector 将 mean contact-F1 从 `0.0344` 提高到 `0.0436`，worst-state F1 从 `0.0122` 提高到 `0.0231`。
- 这是正信号，说明 label-free target contact persistence/contact shell 对真实界面有一定相关性。
- 但提升幅度仍小，不能解决根因。

## warning / error 审查

- optional CA/residue feature warning：符合 no-leak target-only 设置，不影响结果。
- CCD/PDB mirror warning：本阶段读取 tensor dataset 和 generated PDB，不调用需要 mirror 的解析路径，不影响结果。
- 未发现 Traceback/OOM/NaN。

## 当前科学判断

Stage18/20 使候选更物理，但还没有达到“共享 binder 序列兼容多状态 target 且每个 state 形成正确复合物界面”的目标。当前主要瓶颈从 **clash** 转移为 **interface placement 错位**。

下一步不能继续只调 selector，也不能只追 sequence identity。应进入 Stage21：增强 target-relative interface placement 学习。

## Stage21 建议

1. 训练端：提高 `interface_contact_loss`、`interface_distance_loss`、`clash_loss` 对 bb_ca flow 的约束，尤其是 sampled/replay 条件下的约束。
2. 采样端：将 contact-shell/persistence selector 作为默认 safe selector，但只作为筛选，不替代模型学习。
3. 评估端：用 Stage19 evaluator 固定报告 contact-F1、RMSD、clash、persistence，而不是只看 sequence identity。
4. 若短训后 contact-F1 仍低，说明当前多状态 target encoder 没有足够表达正确 hotspot/interface，需要回到数据与 target hotspot construction。
