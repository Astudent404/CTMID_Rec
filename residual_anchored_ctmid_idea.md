# Residual-Anchored Continuous-Time Multimodal Interest Dynamics

## 0. 一句话 Idea

多模态序列推荐中的复杂融合模块容易稀释最强的 ID 协同过滤信号。一个更稳健的建模方式是：保留连续时间多模态兴趣动态作为主干，同时显式加入两条协同残差锚点，一条刻画用户长期 ID 兴趣，一条刻画最近一次交互带来的局部 item 转移，从而让模型同时利用语义多模态信息、时间动态信息和行为协同信号。

## 1. 当前实验发现

最后一轮 Toys and Games 实验中，最强方案不是单纯增加模态、GRU 序列编码或 pairwise rank loss，而是：

```yaml
model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.5
  last_item_residual_weight: 0.5
  lambda_rank: 0.0
```

对应实验组：

```text
toys_6modal_idres0p5_last0p5
```

验证集最优结果：

| Experiment | Best Epoch | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| 6modal + ID residual 0.5 + last-item residual 0.5 | 2 | 0.0171388738 | 0.0101219559 | 0.0079798321 | 0.0239735906 | 0.0118399515 |

关键对比：

| Experiment | NDCG@10 | 结论 |
|---|---:|---|
| 6 模态 baseline | 0.0071135995 | 多模态主干有收益，但仍不强 |
| 6 模态 + ID residual 0.5 | 0.0092682372 | 补回 ID 协同信号显著提升 |
| 6 模态 + ID residual 1.0 | 0.0094952427 | 继续增强 ID residual 仍有效，但收益趋缓 |
| 6 模态 + ID residual 0.5 + last-item residual 0.25 | 0.0097031097 | 最近 item 转移残差继续提升 |
| 6 模态 + ID residual 0.5 + last-item residual 0.5 | 0.0101219559 | 当前最强 |
| 6 模态 + ID residual 0.5 + rank loss 0.1 | 0.0091019640 | rank loss 没有带来收益 |
| 6 模态 + ID residual 0.5 + rank loss 0.3 | 0.0090932433 | rank loss 仍无收益 |

这个结果说明：当前模型的瓶颈不只是“多模态不够”或“序列编码不够复杂”，而是多模态融合后行为协同结构被弱化。最有效的方向是给融合表示增加可解释的协同残差锚点。

## 2. 研究问题

多模态序列推荐通常把 ID、文本、图片、类别、价格、评分等信息融合成统一表示，再用于下一个 item 预测。但在真实推荐场景中，ID 行为序列往往携带最强的协同过滤信号。如果所有模态被过早融合，语义模态可能提升泛化能力，同时也会冲淡 item-to-item 共现、用户历史偏好和最近交互转移。

本文关注的问题是：

**如何在连续时间多模态序列推荐中保留多模态语义与时间动态优势，同时避免融合过程稀释 ID 协同过滤信号？**

进一步拆成三个子问题：

1. 多模态融合是否会削弱 ID 协同信号？
2. ID 协同信号应该作为主干输入、后融合分支，还是残差分支？
3. 用户长期 ID 兴趣和最近 item 转移是否应该分开建模？

## 3. 核心假设

### H1: 多模态融合存在协同信号稀释问题

文本、图片、价格、类别等模态提供语义与属性信息，但它们不一定直接对应用户行为共现结构。复杂融合后，最终表示可能更语义化，却损失 ID embedding 中的局部协同关系。

### H2: ID residual 可以作为长期协同兴趣锚点

用户历史序列经过 ID 模态动态演化后，得到用户在 ID 协同空间中的兴趣状态。该状态与候选 item ID embedding 的点积可以补回长期行为偏好。

### H3: last-item residual 可以作为短期转移锚点

下一个 item 往往强依赖最近一次交互。最近 item 的 ID embedding 与候选 item ID embedding 的点积可以直接捕捉 item-to-item transition，比额外引入 GRU 更简单、更稳定。

### H4: 残差锚点应该与主干共同训练，而不是后处理重排

如果 residual 只在推理阶段加入，会造成训练和评估目标不一致。当前有效方案在 sampled-softmax 训练、single predict 和 full-sort predict 中使用同一套 residual scoring，因此更稳健。

## 4. 方法概述

### 4.1 主干: Continuous-Time Multimodal Interest Dynamics

主干模型保留原 CTMID 设计：

- 多模态 item 表示：`id, category, price, rating, text, image`
- 用户侧模态特定 ODE：建模不同模态兴趣随时间的变化
- item 侧模态特定 ODE：建模 item 表示或流行趋势的时间演化
- 多粒度时间编码：year, month, week, day, interval
- 模态自适应时间权重：不同模态关注不同时间粒度
- time-aligned fusion：将不同模态在目标时间点对齐后融合

主干输出一个融合得分：

```text
s_fuse(u, i, t)
```

它负责捕捉多模态语义、属性、时间动态和跨模态交互。

### 4.2 协同残差锚点 1: ID Interest Residual

从用户历史序列的 ID 模态状态得到用户协同兴趣表示：

```text
h_u^id(t)
```

从候选 item 的原始 ID embedding 得到：

```text
e_i^id
```

计算长期 ID 协同残差：

```text
s_id(u, i, t) = <Norm(h_u^id(t)), Norm(e_i^id)>
```

该分支表示：用户长期行为偏好在 ID 协同空间中是否接近候选 item。

### 4.3 协同残差锚点 2: Last-Item Transition Residual

给定用户最近一次交互 item：

```text
v_last
```

取其 ID embedding：

```text
e_last^id
```

计算最近 item 到候选 item 的转移残差：

```text
s_last(u, i) = <Norm(e_last^id), Norm(e_i^id)>
```

该分支表示：候选 item 是否与用户最近一次行为在协同空间中相近，捕捉短期转移、同类连续浏览、套装/替代/互补等局部行为模式。

### 4.4 最终打分函数

最终预测分数为：

```text
score(u, i, t)
  = s_fuse(u, i, t)
  + alpha * s_id(u, i, t)
  + beta  * s_last(u, i)
  + b_i
```

其中：

- `s_fuse` 是连续时间多模态融合主干得分
- `s_id` 是用户长期 ID 协同兴趣残差
- `s_last` 是最近 item 局部转移残差
- `alpha` 控制长期 ID 协同信号强度
- `beta` 控制最近 item 转移信号强度
- `b_i` 是 item 输出偏置

当前最强实验设置：

```text
alpha = 0.5
beta = 0.5
```

## 5. 方法命名

推荐名称：

**RA-CTMID: Residual-Anchored Continuous-Time Multimodal Interest Dynamics**

中文名：

**残差锚定的连续时间多模态兴趣动态模型**

可以在论文中把两个 residual 称为：

- **Global Collaborative Anchor**：用户长期 ID 兴趣锚点
- **Local Transition Anchor**：最近 item 转移锚点

或者更直接：

- **ID Interest Residual**
- **Last-Item Transition Residual**

## 6. 与已有工作的差异

### vs 普通多模态序列推荐

普通多模态推荐通常关注如何更好地融合文本、图像、类别等信息。RA-CTMID 的核心问题不同：它关注融合之后协同过滤信号被削弱的问题，并通过残差锚点显式保留协同行为结构。

### vs late fusion / ensemble

本方法不是训练后加权多个模型，也不是推理阶段重排。ID residual 和 last-item residual 在训练和评估中同时存在，优化目标一致。

### vs 更复杂的序列编码器

实验中 GRU sequence encoder 没有带来收益。RA-CTMID 说明短期转移不一定需要更复杂的序列模型，直接使用最近 item 的 ID residual 反而更稳定。

### vs item popularity bias

output bias 只刻画 item 全局流行度，无法表达用户相关的协同偏好。ID residual 和 last-item residual 都是 user/item 条件化的残差分支。

### vs pairwise rank loss

rank loss 试图改变优化目标，但实验中没有提升。RA-CTMID 直接改善打分结构，保留被融合稀释的协同成分。

## 7. 预期贡献

1. **提出多模态融合中的协同信号稀释问题**  
   指出多模态语义增强并不总是提升行为排序能力，过度融合可能削弱 ID 协同过滤结构。

2. **提出残差锚定的多模态序列推荐框架**  
   在连续时间多模态主干之外加入协同残差，使模型同时保留语义动态和行为协同。

3. **区分长期兴趣锚点与短期转移锚点**  
   ID interest residual 捕捉用户长期偏好，last-item transition residual 捕捉最近行为转移，两者互补。

4. **保持训练与 full-sort 评估一致**  
   residual 分支在训练、单点预测和 full-sort ranking 中统一启用，避免训练推理不一致。

5. **给出实验证据**  
   在 Toys and Games 上，当前最强配置相对 6 模态 baseline 的 NDCG@10 从 0.00711 提升到 0.01012。

## 8. 论文实验设计

### 8.1 主实验

至少需要在以下数据集验证：

- Toys and Games
- Home and Kitchen
- All Beauty 作为小规模补充或 sanity check

主表报告：

- Recall@10
- NDCG@10
- MRR@10
- Recall@20
- NDCG@20

### 8.2 核心消融

围绕本 idea，最关键的消融应该是：

| Variant | 说明 | 目的 |
|---|---|---|
| CTMID baseline | 无 residual | 验证多模态主干 |
| + ID residual | 只加长期 ID 兴趣残差 | 验证长期协同锚点 |
| + Last residual | 只加最近 item 转移残差 | 验证短期转移锚点 |
| + ID residual + Last residual | 完整 RA-CTMID | 验证两者互补 |
| alpha sweep | alpha = 0.25, 0.5, 0.75, 1.0 | 验证 ID residual 强度 |
| beta sweep | beta = 0.25, 0.5, 0.75, 1.0 | 验证 last residual 强度 |
| learnable gates | alpha/beta 改为可学习门控 | 验证是否优于手动权重 |

### 8.3 与原 CTMID 创新点相关的消融

为了避免论文只像 residual trick，还需要证明连续时间多模态主干仍然有价值：

| Variant | 说明 |
|---|---|
| w/o MS-ODE | 所有模态共享 ODE |
| w/o User-ODE | 移除用户兴趣 ODE |
| w/o Item-ODE | 移除 item ODE |
| w/o MGT | 移除多粒度时间编码 |
| w/o TAF | 移除 time-aligned fusion |
| ExpDecay | 用指数衰减替代 neural ODE |

这些实验用于回答：RA-CTMID 的提升是否来自 residual 与连续时间多模态主干的组合，而不是 residual 单独起作用。

### 8.4 诊断实验

建议增加以下诊断来支撑论文叙事：

| Diagnostic | 目的 |
|---|---|
| 按序列长度分组 | 验证 last residual 是否更适合短序列或稀疏用户 |
| 按 target 与 last item 的时间间隔分组 | 验证最近 item residual 是否主要作用于短间隔行为 |
| 按 item popularity 分组 | 验证 residual 对热门 item 与长尾 item 的影响 |
| 按模态组合分组 | 验证 text/image/price/rating 是否与 residual 互补 |
| residual score 占比分布 | 分析最终分数中主干、ID residual、last residual 的贡献比例 |

## 9. 当前实验结论可以如何写

可以作为论文中的初步观察：

> On Toys and Games, simply adding more modalities improves the baseline, but the largest gain comes from restoring collaborative signals through residual ID-based scoring. The best configuration combines a multimodal continuous-time backbone with both an ID interest residual and a last-item transition residual, indicating that multimodal semantics and collaborative behavior should be modeled as complementary rather than fully entangled signals.

中文表述：

> 在 Toys and Games 上，增加多模态信息本身可以提升性能，但最大增益来自显式补回 ID 协同信号。最优配置同时使用用户 ID 兴趣残差和最近 item 转移残差，说明多模态语义信息与行为协同结构不应被完全混合，而应作为互补信号共同参与排序。

## 10. 风险与需要补强的地方

### 风险 1: 审稿人认为只是简单 residual trick

应对方式：

- 把问题定义为 multimodal fusion dilutes collaborative signals
- 做充分消融，证明长期 ID residual 与短期 last residual 分别有效
- 展示 residual 与 continuous-time multimodal backbone 互补
- 证明不是后处理重排，而是端到端训练的一致打分结构

### 风险 2: last-item residual 可能被认为太接近 item-kNN

应对方式：

- 强调它不是独立 item-kNN，而是作为深度多模态模型的 residual branch
- 加入对比：纯 last-item residual、纯 ID residual、完整模型
- 分析其在短间隔、短序列、局部转移场景中的作用

### 风险 3: 当前只在 Toys 上有强结果

应对方式：

- 优先迁移到 Home and Kitchen
- Home 已有 image features，但配置还需要启用 image
- 至少在两个中大规模数据集上验证同样趋势

### 风险 4: 原始 ODE idea 的贡献可能被 residual 盖住

应对方式：

- 用 RA-CTMID 作为完整模型
- 用 `RA` 和 `CTMID backbone` 分别做消融
- 论文叙事从“单纯模态异步 ODE”调整为“时间动态多模态主干 + 协同残差锚定”

## 11. 推荐下一步实验顺序

1. 在 Toys 上补 `last_item_residual_weight` 单独消融  
   当前有 `id_residual + last residual`，但还需要只开 `last_item_residual_weight` 的实验，证明 beta 分支独立有效。

2. 在 Toys 上做 alpha/beta 网格的小范围搜索  
   推荐组合：

   ```text
   alpha in {0.25, 0.5, 0.75, 1.0}
   beta  in {0.25, 0.5, 0.75, 1.0}
   ```

   可先固定 alpha=0.5，扫 beta。

3. 把 Home and Kitchen 改为 6 模态并复现实验  
   重点验证：

   ```text
   6modal baseline
   6modal + id_residual 0.5
   6modal + id_residual 0.5 + last_item_residual 0.5
   ```

4. 做 CTMID 主干消融  
   验证 ODE、多粒度时间编码、TAF 是否仍有贡献。

5. 增加诊断分析  
   特别是按时间间隔和序列长度分组，解释 last-item residual 为什么有效。

## 12. 可投稿版本标题候选

1. **Residual-Anchored Continuous-Time Multimodal Interest Dynamics for Sequential Recommendation**
2. **Preserving Collaborative Signals in Continuous-Time Multimodal Sequential Recommendation**
3. **RA-CTMID: Collaborative Residual Anchoring for Multimodal Sequential Recommendation**
4. **When Multimodal Fusion Dilutes Collaborative Signals: Residual Anchoring for Sequential Recommendation**

最推荐第 1 个或第 3 个。第 1 个更像正式方法名，第 3 个更简洁。

## 13. 摘要草稿

Multimodal sequential recommendation models integrate item IDs, text, images, categories, prices, and ratings to improve next-item prediction. However, we observe that complex multimodal fusion can dilute the strongest collaborative filtering signals encoded in ID-based behavioral sequences. To address this issue, we propose RA-CTMID, a residual-anchored continuous-time multimodal interest dynamics model. RA-CTMID uses a continuous-time multimodal backbone to capture modality-specific temporal interest evolution, while introducing two collaborative residual anchors: an ID interest residual for long-term user preference and a last-item transition residual for short-term item-to-item dynamics. The final score combines multimodal temporal matching with these residual collaborative signals in a unified training and full-sort evaluation framework. Experiments on Amazon Toys and Games show that the proposed residual anchoring strategy substantially improves ranking performance over the multimodal backbone, suggesting that multimodal semantic signals and collaborative behavioral structures should be modeled as complementary rather than fully entangled factors.

