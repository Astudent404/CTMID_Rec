# RA-CTMID 下一步实验方案

## 0. 当前基线结论

最后一轮 Toys and Games 实验中，当前最强方案是：

```text
toys_6modal_idres0p5_last0p5
```

配置核心：

```yaml
model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.5
  last_item_residual_weight: 0.5
  lambda_rank: 0.0
```

验证集最优结果：

| Metric | Value |
|---|---:|
| Recall@10 | 0.0171388738 |
| NDCG@10 | 0.0101219559 |
| MRR@10 | 0.0079798321 |
| Recall@20 | 0.0239735906 |
| NDCG@20 | 0.0118399515 |

这个结果说明当前最有价值的方向是：

```text
continuous-time multimodal backbone
+ global ID interest residual
+ local last-item transition residual
```

下一步实验的目标不是继续盲目调参，而是构造一条能支撑论文 claim 的证据链。

## 1. 总体实验目标

RA-CTMID 需要证明四件事：

1. **Residual anchoring 是否真的有效**  
   ID residual 和 last-item residual 是否分别有效、是否互补。

2. **Residual 不是简单 trick**  
   它是否优于 output bias、rank loss、GRU encoder、简单 item-kNN 式分支。

3. **Continuous-time multimodal backbone 仍然有贡献**  
   ODE、多粒度时间编码和 time-aligned fusion 是否仍然必要。

4. **结论是否能跨数据集成立**  
   Toys 上有效还不够，至少需要 Home and Kitchen 上复现趋势。

## 2. 实验优先级总览

| Priority | 实验块 | 目的 | 必要性 |
|---:|---|---|---|
| P0 | 复核当前 best 与 baseline 表 | 固定主结果口径 | 必须 |
| P1 | Residual 核心消融 | 证明新 idea 的核心贡献 | 必须 |
| P2 | Alpha/Beta 权重搜索 | 确认 residual 强度和稳定性 | 必须 |
| P3 | CTMID 主干消融 | 防止论文变成 residual trick | 必须 |
| P4 | Home and Kitchen 迁移 | 证明跨数据集有效 | 必须 |
| P5 | 时间与序列诊断 | 支撑机制解释 | 强烈建议 |
| P6 | Baseline 对比 | 构建论文主表 | 后续必须 |
| P7 | Learnable gate 版本 | 提升方法完整性 | 可选增强 |

建议执行顺序：

```text
P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7
```

其中 P0-P4 是写一篇可信方法论文的最低闭环。

## 3. P0: 固定当前主结果

### 3.1 目的

把当前已经跑过的 Toys 实验整理为稳定主结果，确认所有结果来自同一协议：

- same dataset
- same chronological split
- same full-sort evaluation
- same topK
- same seed
- same batch size
- same sampled-softmax negative number
- same early stopping metric

### 3.2 需要整理的结果

从已有日志中整理：

| Group | Config | Best Epoch | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| CTMID-5modal | 5modal baseline | | | | | | |
| CTMID-6modal | 6modal baseline | | | | | | |
| + IDRes 0.25 | `toys_6modal_idres0p25` | | | | | | |
| + IDRes 0.5 | `toys_and_games_ctmid_6modal_idres0p5` | | | | | | |
| + IDRes 0.75 | `toys_6modal_idres0p75` | | | | | | |
| + IDRes 1.0 | `toys_6modal_idres1p0` | | | | | | |
| + IDRes 0.5 + LastRes 0.25 | `toys_6modal_idres0p5_last0p25` | | | | | | |
| + IDRes 0.5 + LastRes 0.5 | `toys_6modal_idres0p5_last0p5` | | | | | | |
| + Rank 0.1 | `toys_6modal_idres0p5_rank0p1` | | | | | | |
| + Rank 0.3 | `toys_6modal_idres0p5_rank0p3` | | | | | | |

### 3.3 输出物

生成一个结果表文档：

```text
ra_ctmid_toys_results_summary.md
```

里面包含：

- 每组 best epoch
- best validation metrics
- 相对 6modal baseline 的提升百分比
- 主要观察结论

## 4. P1: Residual 核心消融

### 4.1 目的

证明 RA-CTMID 的两个 residual 分支分别有效且互补：

- ID interest residual: 长期用户协同偏好
- last-item transition residual: 短期 item 转移

### 4.2 必跑实验

以 Toys and Games 6 模态 CTMID 为主干，保持其他超参不变。

| Experiment | id_residual_weight | last_item_residual_weight | 目的 |
|---|---:|---:|---|
| CTMID-6modal | 0.0 | 0.0 | 主干 baseline |
| LastRes only 0.25 | 0.0 | 0.25 | 验证 last residual 独立贡献 |
| LastRes only 0.5 | 0.0 | 0.5 | 验证 last residual 独立贡献 |
| IDRes only 0.5 | 0.5 | 0.0 | 已有，长期锚点 |
| IDRes only 1.0 | 1.0 | 0.0 | 已有，长期锚点强度 |
| IDRes 0.5 + LastRes 0.25 | 0.5 | 0.25 | 已有，双锚点 |
| IDRes 0.5 + LastRes 0.5 | 0.5 | 0.5 | 已有，当前 best |

当前缺口是：

```text
LastRes only 0.25
LastRes only 0.5
```

这两个实验非常关键。否则无法证明 last-item residual 本身有效，只能说明它在 ID residual 之上有效。

### 4.3 建议新增配置

新增：

```text
configs/experiments/toys_6modal_last0p25.yaml
configs/experiments/toys_6modal_last0p5.yaml
```

配置内容：

```yaml
model:
  id_residual_weight: 0.0
  lambda_rank: 0.0
  last_item_residual_weight: 0.25

train:
  checkpoint_dir: checkpoints/toys_6modal_last0p25
```

```yaml
model:
  id_residual_weight: 0.0
  lambda_rank: 0.0
  last_item_residual_weight: 0.5

train:
  checkpoint_dir: checkpoints/toys_6modal_last0p5
```

### 4.4 预期判断

理想结果：

```text
CTMID-6modal
< LastRes only
< IDRes only
< IDRes + LastRes
```

如果 LastRes only 明显有效，说明最近 item 转移确实是独立信号。

如果 LastRes only 不强，但 IDRes + LastRes 强，说明 last residual 主要依赖 ID interest residual 形成互补，这也可以接受，但论文叙事要写成 complementarity 而不是 independent gain。

## 5. P2: Alpha/Beta 权重搜索

### 5.1 目的

确认当前 `alpha=0.5, beta=0.5` 是否只是偶然，还是在合理范围内稳定最优。

### 5.2 第一阶段: 固定 alpha，扫描 beta

固定：

```text
id_residual_weight = 0.5
```

扫描：

```text
last_item_residual_weight in {0.1, 0.25, 0.5, 0.75, 1.0}
```

已有：

```text
beta = 0.25
beta = 0.5
```

建议新增：

```text
beta = 0.1
beta = 0.75
beta = 1.0
```

对应配置：

```text
toys_6modal_idres0p5_last0p1
toys_6modal_idres0p5_last0p75
toys_6modal_idres0p5_last1p0
```

### 5.3 第二阶段: 固定 beta，扫描 alpha

如果 beta=0.5 仍然最强，固定：

```text
last_item_residual_weight = 0.5
```

扫描：

```text
id_residual_weight in {0.25, 0.5, 0.75, 1.0}
```

建议实验：

```text
toys_6modal_idres0p25_last0p5
toys_6modal_idres0p5_last0p5
toys_6modal_idres0p75_last0p5
toys_6modal_idres1p0_last0p5
```

已有：

```text
toys_6modal_idres0p5_last0p5
```

### 5.4 第三阶段: 小网格确认

如果算力允许，跑完整小网格：

| alpha / beta | 0.25 | 0.5 | 0.75 | 1.0 |
|---:|---:|---:|---:|---:|
| 0.25 | | | | |
| 0.5 | 已有 | 已有 | | |
| 0.75 | | | | |
| 1.0 | | | | |

### 5.5 判断标准

主指标：

```text
NDCG@10
```

辅助指标：

```text
Recall@10
MRR@10
NDCG@20
```

如果某组 NDCG@10 最高但 MRR@10 明显下降，需要谨慎。RA-CTMID 目标是排序质量整体提升，不只是 top20 命中增加。

## 6. P3: CTMID 主干消融

### 6.1 目的

防止论文被质疑为“只是 ID residual + last item residual”。需要证明 continuous-time multimodal backbone 仍然贡献有效。

### 6.2 消融设置

以当前 best 为 base：

```text
6modal + id_residual_weight 0.5 + last_item_residual_weight 0.5
```

在此基础上做主干消融：

| Experiment | 修改 | 目的 |
|---|---|---|
| RA-CTMID full | 无 | 完整模型 |
| w/o MS-ODE | `use_modality_specific_ode: false` | 验证模态特定 ODE |
| w/o User-ODE | `use_user_ode: false` | 验证用户动态 |
| w/o Item-ODE | `use_item_ode: false` | 验证 item 动态 |
| w/o MGT | `use_multi_granularity_time: false` | 验证多粒度时间编码 |
| w/o TAF | `use_time_aligned_fusion: false` | 验证时间对齐融合 |
| ExpDecay | `use_exp_decay: true` | 验证 neural ODE 是否优于简单衰减 |

### 6.3 关键注意

这些消融必须保留 residual：

```yaml
id_residual_weight: 0.5
last_item_residual_weight: 0.5
```

原因是论文最终模型是 RA-CTMID。消融要回答：

```text
在 residual anchoring 已经存在时，CTMID 主干模块是否仍然有贡献？
```

如果直接在无 residual 的 CTMID 上做消融，结论和新 idea 不完全对应。

### 6.4 建议新增配置命名

```text
configs/experiments/toys_ra_full.yaml
configs/experiments/toys_ra_without_ms_ode.yaml
configs/experiments/toys_ra_without_user_ode.yaml
configs/experiments/toys_ra_without_item_ode.yaml
configs/experiments/toys_ra_without_mgt.yaml
configs/experiments/toys_ra_without_taf.yaml
configs/experiments/toys_ra_exp_decay.yaml
```

也可以继续使用已有 `configs/ablations/*.yaml`，但要额外叠加一个 RA 配置，避免忘记 residual。

推荐叠加方式：

```text
base toys_and_games_ctmid.yaml
+ experiments/toys_ra_full.yaml
+ ablations/without_ms_ode.yaml
```

其中 `toys_ra_full.yaml` 负责统一 residual 与 checkpoint 前缀。

## 7. P4: Home and Kitchen 迁移实验

### 7.1 目的

Toys 单数据集不能支撑论文。Home and Kitchen 已经完成 processed 数据和 image features，是最适合的第二个主数据集。

### 7.2 当前配置问题

`configs/home_and_kitchen_ctmid.yaml` 当前仍是：

```yaml
data:
  image_feature_path: null

model:
  modalities: [id, category, price, rating, text]
```

但实际上已有：

```text
data/processed/home_and_kitchen/image_features.npy
```

因此需要新增 Home 6 模态配置，不建议直接覆盖原配置。

### 7.3 建议新增配置

```text
configs/experiments/home_6modal.yaml
configs/experiments/home_6modal_idres0p5.yaml
configs/experiments/home_6modal_idres0p5_last0p5.yaml
```

`home_6modal.yaml`：

```yaml
data:
  image_feature_path: data/processed/home_and_kitchen/image_features.npy

model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.0
  last_item_residual_weight: 0.0
  lambda_rank: 0.0

train:
  checkpoint_dir: checkpoints/home_6modal

eval:
  item_chunk_size: 20000
```

`home_6modal_idres0p5.yaml`：

```yaml
data:
  image_feature_path: data/processed/home_and_kitchen/image_features.npy

model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.5
  last_item_residual_weight: 0.0
  lambda_rank: 0.0

train:
  checkpoint_dir: checkpoints/home_6modal_idres0p5

eval:
  item_chunk_size: 20000
```

`home_6modal_idres0p5_last0p5.yaml`：

```yaml
data:
  image_feature_path: data/processed/home_and_kitchen/image_features.npy

model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.5
  last_item_residual_weight: 0.5
  lambda_rank: 0.0

train:
  checkpoint_dir: checkpoints/home_6modal_idres0p5_last0p5

eval:
  item_chunk_size: 20000
```

### 7.4 Home 最小实验集

Home 数据量很大，优先跑最小闭环：

| Experiment | 目的 |
|---|---|
| Home 5modal current config | 当前默认主干 |
| Home 6modal | 验证 image 模态收益 |
| Home 6modal + IDRes 0.5 | 验证长期协同锚点 |
| Home 6modal + IDRes 0.5 + LastRes 0.5 | 验证完整 RA-CTMID |

如果完整 RA 仍然显著优于 6modal baseline，则跨数据集 claim 基本成立。

### 7.5 Home 的风险

Home item 数约 76 万，full-sort evaluation 成本高。必须保留：

```yaml
eval:
  item_chunk_size: 20000
```

如果显存仍不够，可降低：

```yaml
eval:
  item_chunk_size: 10000
```

不建议为了提速改 sampled evaluation，因为会破坏与 Toys 的口径一致性。

## 8. P5: 诊断实验

### 8.1 目的

诊断实验用于解释 residual 为什么有效，而不只是报告数值提升。

### 8.2 按 target 与 last item 时间间隔分组

计算：

```text
delta_t = target_time - last_interaction_time
```

分组：

| Group | 条件 |
|---|---|
| Very Short | delta_t < 1 day |
| Short | 1 day <= delta_t < 7 days |
| Medium | 7 days <= delta_t < 30 days |
| Long | delta_t >= 30 days |

预期：

- LastRes 在 Very Short / Short 组更有效
- CTMID 时间动态在 Medium / Long 组更有价值
- 完整 RA-CTMID 在各组最稳

需要比较：

```text
6modal baseline
6modal + IDRes
6modal + LastRes
RA-CTMID full
```

### 8.3 按用户序列长度分组

使用训练历史长度或 evaluation prefix 长度：

| Group | 条件 |
|---|---|
| Short history | len <= 5 |
| Medium history | 6 <= len <= 10 |
| Long history | len > 10 |

预期：

- LastRes 对 short history 更重要
- IDRes 对 medium/long history 更稳定
- 多模态主干对 sparse history 也可能有帮助

### 8.4 按 item popularity 分组

按训练集 item 交互次数：

| Group | 条件 |
|---|---|
| Tail | 5-10 interactions |
| Mid | 11-50 interactions |
| Popular | >50 interactions |

预期：

- ID residual 可能更偏 popular item
- 多模态特征可能更帮助 tail/mid item
- 如果 RA-CTMID 在 tail 不下降，就能说明 residual 没有简单强化 popularity bias

### 8.5 按 target item 是否与 last item 同类分组

如果有 category：

```text
same_category = category(target) == category(last_item)
```

分组：

```text
same category
different category
```

预期：

- LastRes 在 same category 中收益更明显
- 多模态主干在 different category 中更重要

### 8.6 residual 分数占比分析

对验证集样本，记录：

```text
s_fuse
alpha * s_id
beta * s_last
output_bias
```

分析：

- top1 item 中各分支平均贡献
- 正样本 rank 改善时哪个分支贡献最大
- 不同 delta_t 分组中的 residual 占比

这个分析可以支撑：

```text
IDRes captures global collaborative preference.
LastRes captures short-term transition.
Backbone captures multimodal temporal matching.
```

## 9. P6: Baseline 对比计划

### 9.1 目的

内部消融证明方法有效，但论文还需要与外部 baseline 比较。

### 9.2 第一阶段 baseline

优先选择实现成本低、结论清楚的方法：

| Method | 类型 | 优先级 |
|---|---|---:|
| GRU4Rec | ID-only sequence | 高 |
| SASRec | ID-only Transformer | 高 |
| BERT4Rec | ID-only bidirectional | 中 |
| TiSASRec | time-aware | 高 |
| TGODE | continuous-time | 高 |
| TGODE-MM | continuous-time + multimodal | 高 |

### 9.3 第二阶段 baseline

多模态相关：

| Method | 类型 | 优先级 |
|---|---|---:|
| UniSRec | text-enhanced SR | 中 |
| MISSRec | multimodal SR | 高 |
| HM4SR | time-aware multimodal | 高 |
| MTSTRec | time-aligned multimodal | 高 |

### 9.4 Baseline 口径要求

所有 baseline 尽量使用：

```text
same chronological split
full-sort ranking
mask history
topK = [5, 10, 20]
```

如果某些 baseline 只能 sampled evaluation，必须单独标注，不与 full-sort 结果直接混比。

## 10. P7: Learnable Gate 版本

### 10.1 目的

当前 `alpha` 和 `beta` 是手动超参。为了方法更完整，可以设计 learnable gate：

```text
score = s_fuse
      + g_id(u, t) * s_id
      + g_last(u, t) * s_last
      + b_i
```

其中：

```text
g_id, g_last in [0, 1]
```

### 10.2 简单版本

全局可学习参数：

```text
alpha = sigmoid(a)
beta = sigmoid(b)
```

优点：

- 实现简单
- 比固定 0.5 更优雅

缺点：

- 表达能力有限

### 10.3 用户条件化版本

根据用户序列输出生成 gate：

```text
[g_id, g_last] = sigmoid(MLP(seq_output))
```

优点：

- 可根据用户状态动态决定依赖长期兴趣还是最近转移

缺点：

- 可能过拟合
- 需要额外消融

### 10.4 时间条件化版本

根据 `delta_t` 生成 gate：

```text
g_last = sigmoid(MLP(log1p(delta_t)))
```

预期：

- 短间隔时 g_last 更大
- 长间隔时 g_last 更小

这个版本和论文叙事最匹配，但需要更细的诊断验证。

### 10.5 是否现在做

建议在 P1-P5 完成后再做。  
如果固定 `alpha=0.5, beta=0.5` 已经非常强，可以把 learnable gate 放到 future work 或 appendix。

## 11. 推荐实验批次

### Batch A: Toys residual 补洞

目的：补齐新 idea 最核心证据。

实验：

```text
toys_6modal_last0p25
toys_6modal_last0p5
toys_6modal_idres0p5_last0p1
toys_6modal_idres0p5_last0p75
toys_6modal_idres0p5_last1p0
```

产出：

- residual ablation 表
- beta sweep 曲线

### Batch B: Toys RA 主干消融

目的：证明 CTMID 主干仍然有贡献。

实验：

```text
toys_ra_without_ms_ode
toys_ra_without_user_ode
toys_ra_without_item_ode
toys_ra_without_mgt
toys_ra_without_taf
toys_ra_exp_decay
```

产出：

- backbone ablation 表

### Batch C: Home 迁移

目的：验证跨数据集有效性。

实验：

```text
home_6modal
home_6modal_idres0p5
home_6modal_idres0p5_last0p5
```

如果时间允许，再加：

```text
home_6modal_last0p5
home_6modal_idres0p5_last0p25
```

产出：

- Home 主结果表
- 跨数据集趋势比较

### Batch D: 诊断分析

目的：写论文分析部分。

分析：

```text
delta_t group
sequence length group
item popularity group
same-category transition group
score component contribution
```

产出：

- 2-3 张诊断表
- 1 张 residual contribution 图

## 12. 最小可发表实验闭环

如果时间有限，最低限度需要完成：

1. Toys residual 核心消融  
   包括 baseline、IDRes only、LastRes only、IDRes+LastRes。

2. Toys backbone 消融  
   至少 w/o MS-ODE、w/o User-ODE、w/o Item-ODE、w/o MGT、w/o TAF。

3. Home 迁移  
   至少 6modal baseline、+IDRes、+IDRes+LastRes。

4. 一个诊断实验  
   推荐 delta_t 分组，因为它最能解释 last residual 和 continuous-time backbone 的分工。

5. 至少 3 个外部 baseline  
   推荐 GRU4Rec、SASRec、TiSASRec 或 TGODE。

## 13. 论文图表规划

### Table 1: Overall Performance

数据集：

```text
Toys and Games
Home and Kitchen
```

方法：

```text
GRU4Rec
SASRec
TiSASRec
TGODE
CTMID
RA-CTMID
```

指标：

```text
Recall@10
NDCG@10
MRR@10
Recall@20
NDCG@20
```

### Table 2: Residual Ablation

列：

```text
Backbone
IDRes
LastRes
Recall@10
NDCG@10
MRR@10
```

### Table 3: Backbone Ablation

列：

```text
Variant
Recall@10
NDCG@10
Relative Drop
```

### Figure 1: Alpha/Beta Sensitivity

展示：

```text
NDCG@10 vs beta when alpha=0.5
NDCG@10 vs alpha when beta=0.5
```

### Figure 2: Delta-t Group Performance

展示：

```text
Very Short / Short / Medium / Long
```

比较：

```text
CTMID
IDRes only
LastRes only
RA-CTMID
```

### Figure 3: Score Component Contribution

展示：

```text
s_fuse
alpha * s_id
beta * s_last
output_bias
```

按不同 delta_t 或 sequence length 分组。

## 14. 结果解释模板

如果实验符合预期，可以这样组织论文结论：

1. 6 模态 CTMID 优于低模态模型，说明多模态和时间动态主干有效。
2. ID residual 明显提升，说明融合主干确实会稀释 ID 协同信号。
3. Last-item residual 进一步提升，说明短期 item transition 是独立于长期用户兴趣的重要信号。
4. Rank loss 和 GRU encoder 没有带来收益，说明改造打分结构比增加优化复杂度或序列复杂度更有效。
5. 主干消融下降，说明 residual anchoring 需要和 continuous-time multimodal backbone 结合，而不是替代主干。
6. Home 上趋势一致，说明方法不是 Toys 特例。
7. delta_t 诊断显示 last residual 更偏短间隔，CTMID 主干更偏长间隔，二者分工合理。

## 15. 当前最建议马上做的三件事

1. **补 LastRes only 实验**  
   这是当前最大证据缺口。

2. **在 best 配置上做 backbone 消融**  
   证明 RA-CTMID 不是只有 residual 起作用。

3. **把 Home 开成 6 模态并跑三组主实验**  
   证明跨数据集有效。

完成这三件事后，这个 idea 才从“一个有效实验技巧”变成“可以支撑论文方法的实验故事”。

