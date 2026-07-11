# WM Evaluation Docs

Paper-facing result locations are organized in
[paper_data_catalog.md](paper_data_catalog.md). Use that catalog first for
main-paper RL results, appendix dataset/lambda ablations, dataset validity
status, and raw artifact locations.

The older notes below are project-specific working notes.

# Dreamer WM Checkpoint Evaluation（简版）

一句话：`context=5` 用于用真实轨迹做 latent 热启动（bootstrap），然后再做 horizon 的 open-loop 生成；`context` 里的 5 帧不计入待评估 horizon。  

目标：评估不同 `Dreamer WM` checkpoint 的有效性。  

1) zero-shot RL  
- 评估协议与 [LYK-love/rl-in-pixel-env](https://github.com/LYK-love/rl-in-pixel-env) 保持一致。  
- 结论：`reproduction baseline` 与 `w=0` 均很差，`w=0.01` 最好。  
- 表格与 size400m 补充结果见 [zero_shot_mbrl.md](zero_shot_mbrl.md)。  

2) Generated-video ball tracking（定量）  
- 方法与结果详见 [generated_video_ball_tracking.md](/scorpio/home/luyukuan/projects/dreamerv3-reborn/docs/dreamer_wm_experiment_results/generated_video_ball_tracking.md)。  
- 外部 WM 对照见 [external_wm_sam2_ball_tracking.md](/scorpio/home/luyukuan/projects/dreamerv3-reborn/docs/dreamer_wm_experiment_results/external_wm_sam2_ball_tracking.md)。  

3) Short-horizon prediction（定量）
- 方法：同一 replay windows、同一 context（5 帧）、同一未来 action 序列，做 open-loop 生成并与 GT 对齐计算 MSE。  
- 指标：全局 pixel MSE，Ball / Left paddle / Right paddle masked MSE。  
- 结论（按结果归档）：  
  - size200m：`size200m_mask1_w1` 在 ball 指标上多数 H 最优，但不同 H 全局/对象排序会变化。  
  - size400m：`size400m_mask1_w01/w1` 在多个 H 对象指标上有优势；`w0`/`w0.01` 在全局指标上未始终统治。  

### size200m short-horizon MSE（5 way）

### Context 与 horizon 的关系
- 这两个实验都设 `context=5`，但它不是要预测的步数。  
- 先喂真实序列 `o1..o5`（以及对应动作）给 WM，执行 `bootstrap` 得到对齐后的内部 latent；  
- 然后从 `t=6` 开始用动作 `a5` 预测 `\hat{o6}`，接着 open-loop 继续预测，`\hat{o7}`、`\hat{o8}` …，一直到 horizon 长度。  
- 所以 `context` 是“对齐状态”的预热，不是统计误差对齐里的 horizon 帧。  

#### H=12
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size200m_repro` | 0.002983 | 0.103720 | 0.052416 | 0.046280 |
| `size200m_mask1_w0` | 0.001300 | 0.091289 | 0.025814 | 0.022030 |
| `size200m_mask1_w001` | 0.000885 | 0.086355 | 0.018029 | 0.008852 |
| `size200m_mask1_w01` | 0.001390 | 0.066053 | 0.012604 | 0.032006 |
| `size200m_mask1_w1` | 0.000958 | 0.026249 | 0.009603 | 0.028844 |

#### H=32
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size200m_repro` | 0.002977 | 0.099002 | 0.049867 | 0.061387 |
| `size200m_mask1_w0` | 0.001588 | 0.090174 | 0.030575 | 0.017878 |
| `size200m_mask1_w001` | 0.001146 | 0.064931 | 0.015170 | 0.010699 |
| `size200m_mask1_w01` | 0.001239 | 0.047499 | 0.014834 | 0.010612 |
| `size200m_mask1_w1` | 0.001595 | 0.044591 | 0.019814 | 0.041176 |

#### H=48
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size200m_repro` | 0.002972 | 0.101148 | 0.049378 | 0.034817 |
| `size200m_mask1_w0` | 0.001475 | 0.095421 | 0.024097 | 0.004160 |
| `size200m_mask1_w001` | 0.001363 | 0.071390 | 0.019069 | 0.009745 |
| `size200m_mask1_w01` | 0.001217 | 0.050592 | 0.017321 | 0.013351 |
| `size200m_mask1_w1` | 0.001608 | 0.045613 | 0.017715 | 0.021200 |

### size400m short-horizon MSE（4 way）

#### H=12
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size400m_mask1_w0` | 0.000834 | 0.069385 | 0.013331 | 0.018912 |
| `size400m_mask1_w001` | 0.000911 | 0.065223 | 0.012450 | 0.016595 |
| `size400m_mask1_w01` | 0.001296 | 0.066134 | 0.010319 | 0.029268 |
| `size400m_mask1_w1` | 0.000652 | 0.025408 | 0.007342 | 0.020204 |

#### H=32
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size400m_mask1_w0` | 0.001291 | 0.070485 | 0.024635 | 0.020235 |
| `size400m_mask1_w001` | 0.001242 | 0.073688 | 0.012776 | 0.015636 |
| `size400m_mask1_w01` | 0.001032 | 0.051508 | 0.017623 | 0.005433 |
| `size400m_mask1_w1` | 0.001004 | 0.047326 | 0.014229 | 0.020391 |

#### H=48
| model | global_mse | ball_mse | left_paddle_mse | right_paddle_mse |
| --- | ---: | ---: | ---: | ---: |
| `size400m_mask1_w0` | 0.001281 | 0.099546 | 0.019656 | 0.018368 |
| `size400m_mask1_w001` | 0.001060 | 0.073205 | 0.015199 | 0.004771 |
| `size400m_mask1_w01` | 0.001077 | 0.060633 | 0.012249 | 0.009435 |
| `size400m_mask1_w1` | 0.001151 | 0.039948 | 0.012409 | 0.014329 |

实现说明（online训练）：交互数据即时入 replay，形成可训序列后优先被 online FIFO 消费；replay 采样仍为默认均匀采样。  

对应 artifact：  
- [size200m_h12/32/48](/scorpio/home/luyukuan/projects/dreamerv3-reborn/notebook_outputs/pong_wm_visual_size200m_mask1_five_way_h12_20260601)  
- [size400m_h12/32/48](/scorpio/home/luyukuan/projects/dreamerv3-reborn/notebook_outputs/pong_wm_visual_size400m_mask1_four_way_h12_20260601)  
