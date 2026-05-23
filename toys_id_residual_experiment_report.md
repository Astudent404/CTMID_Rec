# Toys 6 模态 ID Residual 实验记录

日期：2026-05-18

## 背景

本轮实验目标是在 Toys and Games 数据集上提升 6 模态 CTMID 的排序指标，尤其是 `NDCG@10`。

此前已经完成的 6 模态配置包含：

- `id`
- `category`
- `price`
- `rating`
- `text`
- `image`

6 模态 baseline 相比 5 模态已有提升，但指标仍偏低。进一步尝试过 `pop-bias / 去重负采样` 和 `GRU sequence encoder`，两者在 Toys 上没有带来收益，甚至出现反效果。因此本次成功改动聚焦在一个更保守的方向：保留多模态融合主干，同时显式补回 ID 协同过滤信号。

## 成功改动

核心思路：在原有多模态融合打分之外，额外加入一条 ID residual scoring 分支。

最终打分形式：

```text
score(u, i) = fused_score(u, i) + alpha * id_score(u, i) + output_bias_i
```

其中：

- `fused_score(u, i)` 是 CTMID 原本的多模态融合打分。
- `id_score(u, i)` 使用用户侧演化后的 `id` 模态状态与 item 原始 ID embedding 计算点积。
- `alpha` 由配置项 `id_residual_weight` 控制。
- 本次有效配置使用 `alpha = 0.5`。

这个改动的动机是：多模态融合可能会稀释 ID embedding 中最强的协同过滤信号；ID residual 可以让模型在保留多模态建模能力的同时，直接利用 ID 行为序列信号。

## 代码实现

主要改动文件：

- `ctmid/model/ctmid.py`
- `configs/toys_and_games_ctmid.yaml`

`ctmid/model/ctmid.py` 中新增了以下逻辑：

- 新增 `id_residual_weight` 配置读取。
- 新增 `use_id_residual` 开关，只有当 `id_residual_weight > 0` 且存在 `id` 模态时启用。
- 新增 `id_residual_norm`，对 ID residual 分支的 item 表征做归一化。
- 新增 `_id_item_representations()`，用于取得 item 的原始 ID embedding 表征。
- 新增 `_id_residual_logits()`，用于计算用户 ID 状态和 item ID 表征之间的 residual logits。
- 在 `calculate_loss()` 中，将 ID residual logits 加到 sampled-softmax 训练 logits 上。
- 在 `predict()` 中，将 ID residual score 加到单点预测分数上。
- 在 `full_sort_predict()` 中，将 ID residual score 加到全量排序分数上。

配置文件中本次有效设置为：

```yaml
model:
  sequence_encoder: none
  id_residual_weight: 0.5

training:
  checkpoint_dir: checkpoints/toys_and_games_ctmid_6modal_idres0p5
```

注意：本次成功结果不依赖 GRU sequence encoder。`sequence_encoder` 保持为 `none`。

## 实验设置

数据集：`toys_and_games`

模态：

```text
id, category, price, rating, text, image
```

训练关键设置：

- sampled-softmax 负采样数量：`4096`
- batch size：`2048`
- 验证方式：full-sort ranking
- 主验证指标：`NDCG@10`

日志与 checkpoint：

- 训练日志：`log/train_toys_6modal_idres0p5.log`
- 最优 checkpoint：`checkpoints/toys_and_games_ctmid_6modal_idres0p5/best.pt`

## 结果

最优 epoch：`6`

最优验证指标：

| Metric | Value |
| --- | ---: |
| Hit@5 / Recall@5 | 0.0112683354 |
| NDCG@5 | 0.0078242810 |
| MRR@5 | 0.0066890218 |
| Hit@10 / Recall@10 | 0.0157500294 |
| NDCG@10 | 0.0092682372 |
| MRR@10 | 0.0072817237 |
| Hit@20 / Recall@20 | 0.0215591381 |
| NDCG@20 | 0.0107301448 |
| MRR@20 | 0.0076794062 |

与已有实验对比：

| Experiment | Best Epoch | NDCG@10 | 备注 |
| --- | ---: | ---: | --- |
| 5 模态 baseline | 6 | 0.0052868105 | 未启用 image |
| 6 模态 baseline | 17 | 0.0071135995 | 启用 image |
| 6 模态 + GRU sequence encoder | 7 | 0.0059421432 | 反效果 |
| 6 模态 + ID residual, alpha=0.5 | 6 | 0.0092682372 | 本次成功结果 |

相对提升：

| 对比对象 | NDCG@10 提升 |
| --- | ---: |
| 相对 6 模态 baseline | +30.29% |
| 相对 5 模态 baseline | +75.30% |

相对 6 模态 baseline 的其他指标变化：

| Metric | Baseline | ID Residual | Relative Change |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.0126732049 | 0.0157500294 | +24.27% |
| NDCG@10 | 0.0071135995 | 0.0092682372 | +30.29% |
| MRR@10 | 0.0054262354 | 0.0072817237 | +34.20% |

## 训练走势

本次训练在 epoch 6 达到最优，之后没有继续刷新最优验证指标，并最终 early stopping。

关键 epoch 的 `NDCG@10`：

| Epoch | NDCG@10 |
| ---: | ---: |
| 1 | 0.003834 |
| 2 | 0.006496 |
| 3 | 0.008458 |
| 4 | 0.008884 |
| 5 | 0.009109 |
| 6 | 0.009268 |
| 7 | 0.009115 |
| 8 | 0.009158 |
| 9 | 0.009017 |
| 10 | 0.009018 |
| 11 | 0.008972 |
| 12 | 0.008790 |
| 13 | 0.008954 |
| 16 | 0.008841 |

## 结论

这次改动是有效的。它说明当前 CTMID 在 Toys 数据集上的主要问题之一不是缺少复杂序列编码，而是多模态融合后 ID 协同过滤信号被削弱。

ID residual 的优势是改动小、风险低、解释直接：

- 不改变数据集划分。
- 不改变 full-sort 评估口径。
- 不改变负采样训练框架。
- 不引入额外序列模型复杂度。
- 训练和推理路径保持一致，sampled-softmax、单点预测和全量排序均启用同一套 residual score。

## 后续建议

下一步建议优先做低成本消融：

- 在 Toys 上测试 `id_residual_weight = 0.25, 0.75, 1.0`，确认最优 residual 强度。
- 将同样改动迁移到 Home 数据集，验证是否具有跨数据集稳定性。
- 如果 residual 在多个数据集有效，再考虑把 `id_residual_weight` 做成 learnable gate，而不是手动超参。
- 暂时不要继续推进 GRU sequence encoder，除非先重新设计验证它确实能补充而不是扰乱当前用户状态建模。

