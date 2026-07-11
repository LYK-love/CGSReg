# Dyna Freeze-WM Claim 诊断报告

本文要验证一个 claim：Dyna-style MBRL 的高分是否主要来自 world model（WM）持续跟随新数据，而不是来自 WM 本身的稳健生成能力。

实验细节见 `CGSReg/experiments/freeze_wm_diagnostic/README.md` 和 `conventions.md`。本文只保留和 claim 相关的信息。核心指标是 W&B 上真实环境评估的 `eval_real/score_mean`。

## 1. Claim

Dyna-style MBRL 的训练闭环是：

```text
policy 收集真实环境数据 -> 更新 WM -> WM 生成 imagined rollout -> 更新 policy
```

这个闭环可能掩盖一种失败模式：

- WM 只是在当前 policy 数据附近做扩增；
- WM 的 zero-shot 生成能力不强；
- 但 WM 一直被最新数据修正，所以 Dyna-style RL 仍然能拿到高分。

因此我原本的 claim 是：

> 如果 Dyna-style 的效果主要来自 WM 持续在线更新，那么冻结 WM 后，后续 policy learning 应该明显退化。

冻结 WM 指：停止 WM optimizer updates，但继续 policy/value training、real-env collection 和 real-env evaluation。冻结后，policy 只能依赖一个固定 WM 继续优化，因此这个阶段更接近 fixed-model / zero-shot MBRL。

## 2. 实验设计

每个项目跑同一组干预：

| Condition | Freeze point | 训练到 | 作用 |
| --- | ---: | ---: | --- |
| no-freeze | none | 1.5T | 对照组，WM 全程更新 |
| f0.5 | 0.5T | 1.5T | 早期冻结 WM |
| f0.75 | 0.75T | 1.5T | 中期冻结 WM |
| f1.0 | 1.0T | 1.5T | 原预算结束后冻结 WM，再训练 0.5T |

比较方式：

| 项 | 设置 |
| --- | --- |
| 环境 | ALE Pong |
| 主要指标 | `eval_real/score_mean` |
| 横轴 | training progress / 原始 T |
| 判据 | freeze 后真实环境得分是否相对 no-freeze 明显下降 |

每个项目的 T 都是该项目的实验复现所需的训练量预算，五个项目的 T 单位不同，但都映射到同一个相对进度：

| Project | T | W&B project |
| --- | ---: | --- |
| DreamerV3 | 100,000 env steps | `dreamer-dyna-freeze-wm` |
| STORM | 100,000 sample steps | `storm-dyna-freeze-wm` |
| Simulus | 600 epochs | `simulus-dyna-freeze-wm` |
| TWISTER | 100,000 model/env steps | `twister-dyna-freeze-wm` |
| DIAMOND | 100,000 collected train-env steps | `diamond-dyna-freeze-wm` |

## 3. 结果

| Project | 观察 | 对 claim 的影响 |
| --- | --- | --- |
| DreamerV3 | 0.5T、0.75T、1.0T freeze 后，`eval_real/score_mean` 都明显低于 no-freeze | 支持 claim |
| STORM | freeze 后没有明显掉分，也没有回退到 zero-shot RL 的低分状态 | 反驳 claim |
| Simulus | freeze 和 no-freeze 之间没有稳定差距 | 反驳 claim |
| TWISTER | 1.0T freeze 明显退化；0.75T freeze 中后期退化；0.5T freeze 先降后恢复 | 部分支持，但不干净 |
| DIAMOND | 未完成，训练成本过高 | 无结论 |

结论很直接：原始 claim 不是跨项目成立的。它解释 DreamerV3，但解释不了 STORM 和 Simulus。TWISTER 提供部分支持，但不是“冻结即崩”的干净证据。

## 4. 分项目证据

### DreamerV3

W&B: https://wandb.ai/ssl-lab/dreamer-dyna-freeze-wm

![image-20260618173429018](/Users/lyk/Library/Application Support/typora-user-images/image-20260618173429018.png)

DreamerV3 是最符合 claim 的结果。三个 freeze 点之后，真实环境分数都明显下降。固定 WM 不能支撑后续 policy learning，说明该设置下 DreamerV3 强依赖 WM 持续更新。

### STORM

W&B: https://wandb.ai/ssl-lab/storm-dyna-freeze-wm

![image-20260619133523100](/Users/lyk/Library/Application Support/typora-user-images/image-20260619133523100.png)

STORM 是主要反例。freeze 后 `eval_real/score_mean` 没有明显下降。

这和另一个观察冲突：zero-shot MBRL 实验里，STORM 的 WM 生成能力极差，zero-shot RL 接近 0 分。按原始 claim，freeze 后应该回退到类似 zero-shot 的低分状态，但实际没有。

### Simulus

W&B: https://wandb.ai/ssl-lab/simulus-dyna-freeze-wm/

![image-20260618173129101](/Users/lyk/Library/Application Support/typora-user-images/image-20260618173129101.png)

Simulus 也不支持 claim。`eval_real/score_mean` 看不出 freeze 造成稳定退化。也就是说，至少这个指标没有显示 Simulus 的 Dyna-style 表现主要依赖 WM 持续更新。

### TWISTER

W&B: https://wandb.ai/ssl-lab/twister-dyna-freeze-wm

TWISTER 当前是单 seed、未完成观察：

| Condition | 观察 | 判断 |
| --- | --- | --- |
| no-freeze | 50k 后持续上升；100k 后基本满分；最后约 21.0 | 对照正常 |
| f1.0 | 100k freeze 后从约 18 降到 150k 的约 8.6/8.8 | 支持晚期 freeze 退化 |
| f0.75 | 75k freeze 后先升到局部峰值，随后降到约 8.4/7.4 | 支持中后期退化 |
| f0.5 | 50k freeze 后先降，但 100k 后恢复到 18 到 19 | 不支持持续退化 |

TWISTER 说明持续更新 WM 可能有帮助，尤其 late freeze 会损害后续学习。但它不支持“WM 一停止训练，RL 就立刻持续崩掉”。

### DIAMOND

DIAMOND 没有形成可用结论。当前 Dyna loop 的 actor-critic training 成本过高，实验已停止。

## 5. 结论和疑问

### 结论

| 命题 | 当前判断 |
| --- | --- |
| Dyna-style 高分可能掩盖 WM 生成能力不足 | DreamerV3 支持 |
| freeze WM 后 policy learning 应普遍退化 | 被 STORM 和 Simulus 反驳 |
| freeze 实验可以直接等价于 zero-shot MBRL 能力测试 | 不成立，至少 STORM 反例明显 |

更准确的说法是：

> Freeze-WM 诊断能发现某些算法对 WM 持续更新的依赖，例如 DreamerV3；但 freeze 后不退化并不必然说明 WM 有强 zero-shot 生成能力。

### 主要疑问

最大的疑问是 STORM：

> 为什么 STORM 的 zero-shot MBRL 接近 0 分，但 freeze WM 后不退化？

可能解释：

| 解释 | 含义 | 需要补的证据 |
| --- | --- | --- |
| **局部分布足够** | **freeze 时 WM 已覆盖当前 policy 附近分布，能支持局部改进，但不能从头 zero-shot** | freeze 后 policy state-action 分布漂移 |

下一步应比较 freeze 后的 policy 分布漂移、WM rollout error、imagined return 与 real return 的一致性，而不是只看最终真实环境分数。
