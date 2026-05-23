# Idea 5 详细实验方案：Continuous-Time Multimodal Interest Dynamics for Sequential Recommendation

## 0. 实验目标

本实验方案围绕 `polished_idea_5.md` 中的核心 idea：

> Continuous-Time Multimodal Interest Dynamics for Sequential Recommendation

核心假设是：

1. 不同模态具有不同信息生命周期，例如 image 慢变、text 中速变化、price/promotion 快变；
2. 用户兴趣在不同模态维度上的演化速度不同；
3. item 全局分布或流行趋势也在不同模态维度上异步演化；
4. 因此，多模态序列推荐不应只做“时间 embedding + modality fusion”，而应显式建模 **modality-specific continuous-time dynamics**。

实验需要回答以下问题：

- **RQ1：整体效果**：提出的方法是否优于经典 SR、time-aware SR、continuous-time SR、多模态 SR 和最新 multimodal temporal baselines？
- **RQ2：核心模块有效性**：模态特定 ODE、用户兴趣 ODE、item 分布 ODE、多粒度时间编码、跨模态时间对齐融合分别是否有效？
- **RQ3：模态异步假设是否成立**：模型是否真的学习到了不同模态不同演化速度？
- **RQ4：在哪些场景提升最大**：长时间间隔、高 item 流行度波动、价格敏感、冷启动、季节性场景是否提升更明显？
- **RQ5：效率代价是否可接受**：ODE solver 的训练/推理开销与性能收益之间的 trade-off 如何？

---

## 1. 推荐方法命名

建议暂定方法名：

**CTMID**：Continuous-Time Multimodal Interest Dynamics

或更强调异步模态：

**A-MODE**：Asynchronous Multimodal ODE for Sequential Recommendation

文中可以统一写作：

> We propose CTMID, a modality-specific continuous-time framework for asynchronous multimodal interest dynamics.

---

## 2. 数据集设计

实验建议分成 **主实验数据集** 和 **扩展验证数据集**。

### 2.1 主实验数据集：Amazon Reviews 5-core

主实验建议使用 Amazon Reviews 系列，因为 TGODE、RoTE、HM4SR 都使用 Amazon 数据，便于公平比较。

| Dataset | Users | Items | Interactions | 推荐理由 | 可用模态 |
|---|---:|---:|---:|---|---|
| Amazon Beauty | 22,363 | 12,101 | ~198k | HM4SR/TGODE/RoTE 都使用；图像与文本对购买偏好重要 | ID, image, title/description/category/brand, price, timestamp |
| Amazon Toys and Games | 19,412 | 11,924 | ~167k | 季节性明显，适合验证时间动态 | ID, image, text, category, price, timestamp |
| Amazon Sports and Outdoors | 35,598 | 18,357 | ~296k | 时间跨度较大，用户间隔不规则明显 | ID, image, text, category, price, timestamp |
| Amazon Video Games / Games | 24,303 | 10,672 | ~231k | 产品生命周期和热度变化明显；适合 item trend ODE | ID, image, text, category, price, timestamp |
| Amazon Home and Kitchen | 66,520 | 28,238 | ~552k | HM4SR 使用；规模更大，用于验证泛化和效率 | ID, image, text, category, price, timestamp |

### 2.2 推荐主实验组合

为了控制实验规模，建议主实验先使用 4 个数据集：

1. **Beauty**：标准 benchmark；
2. **Toys and Games**：季节性强；
3. **Sports and Outdoors**：时间间隔跨度大；
4. **Home and Kitchen**：规模更大，验证泛化。

如果计算资源有限，第一阶段可以只跑：

- Beauty
- Toys and Games
- Sports and Outdoors

这三个数据集与 RoTE/TGODE/HM4SR 的交集最大。

### 2.3 扩展验证数据集：H&M Personalized Fashion Recommendations

H&M 很适合验证“视觉慢变 + 价格/季节快变”的假设。

| Dataset | 推荐理由 | 可用模态 |
|---|---|---|
| H&M Personalized Fashion Recommendations | 有交易时间、商品图像、商品属性、价格、季节性；服饰推荐中图像风格和季节变化明显 | image, product name, product type, color, department, price, timestamp |

建议用法：

- 作为扩展数据集，而不是第一主实验；
- 可以使用 full dataset 或参考 MTSTRec 的 trousers subset；
- 如果 full dataset 太大，先选择 1-2 个大类，例如 trousers、dresses、tops。

### 2.4 可选高时效数据集

如果后续想更强地证明“不同模态生命周期”，可以考虑新闻/短视频类数据，但实现成本更高。

| Dataset | 推荐理由 | 风险 |
|---|---|---|
| MIND News | 新闻标题/摘要时效性强，category/topic 变化快 | 任务更偏 news recommendation，不完全是 item SR |
| KuaiRec / KuaiRand / MicroLens | 短视频推荐，热点标签/视频内容时效性强 | 多模态特征获取和 baseline 对齐更复杂 |

建议：**第一篇论文不必强行引入这些数据集**。Amazon + H&M 已经足够支撑主线。

---

## 3. 数据预处理方案

### 3.1 基础过滤

参考 HM4SR、RoTE、TGODE：

1. 移除重复交互；
2. 按 user 内 timestamp 升序排序；
3. 使用 5-core 过滤：每个 user 和 item 至少 5 次交互；
4. 最大序列长度设为 50（Amazon）；H&M 可设为 20 或 50；
5. 短序列 padding，长序列截断最近 `max_len` 个 item。

### 3.2 模态特征构造

#### 3.2.1 ID 模态

- 可学习 item embedding；
- hidden size 建议统一为 64 或 128。

#### 3.2.2 Text 模态

Amazon：

- 拼接 `title + category + brand + description`；
- 使用 frozen BERT / Sentence-BERT / MiniLM 编码；
- 若计算资源允许，可使用 Llama / text-embedding 模型离线提取，但为了可复现，建议主实验用 BERT-base 或 sentence-transformers。

H&M：

- 拼接 `prod_name + product_type_name + product_group_name + graphical_appearance_name + colour_group_name + department_name + detail_desc`。

#### 3.2.3 Image 模态

- 使用商品第一张图；
- 特征提取器：ViT-B/16 或 ResNet-50；
- frozen feature + linear projection 到 hidden size；
- 如果图片缺失，用 zero vector 或 learnable missing-image token。

#### 3.2.4 Price 模态

Amazon：

- metadata price 若存在则使用；
- 缺失 price 使用 category median 或 missing-price token；
- price 做 log transform：`log(price + 1)`，再 z-score normalization。

H&M：

- 使用 transaction price 或 article price；
- transaction price 更适合验证价格时间波动。

#### 3.2.5 Category 模态

- 多级 category 可用 embedding；
- 可将 category 作为独立模态，也可并入 text；
- 建议主实验中保留 category 作为单独模态，以观察 category 的中低频时间变化。

### 3.3 时间特征构造

对每个 timestamp 生成：

1. absolute time：Unix timestamp；
2. time interval：`Δt_i = t_i - t_{i-1}`；
3. calendar features：year, month, day, week, day-of-week, hour；
4. relative position：序列位置；
5. time bucket：用于分组分析。

多粒度时间编码建议：

- Amazon：year / month / day；
- H&M：month / week / day；
- 如果 dataset 有小时级：day / hour / minute。

---

## 4. 数据划分与评估协议

建议设置两个评估协议：**主协议使用全局时间切分**，**补充协议使用 leave-one-out**。

### 4.1 Protocol A：Global Chronological Split（主协议）

参考 TGODE，更符合真实推荐场景。

做法：

1. 按所有交互的 timestamp 全局排序；
2. 前 80% 作为 train；
3. 中间 10% 作为 validation；
4. 最后 10% 作为 test；
5. 对 validation/test 中每个目标交互 `(u, v, t)`，只使用用户在 `t` 之前的历史序列作为输入。

优点：

- 避免用未来交互训练过去预测；
- 更适合验证 item distribution ODE；
- 更适合证明 temporal generalization。

缺点：

- 与 HM4SR/RoTE 的 leave-one-out 设置不完全一致。

因此建议作为 **main evaluation**。

### 4.2 Protocol B：Leave-One-Out Split（补充协议）

参考 HM4SR、RoTE。

做法：

1. 每个 user 的最后一个交互作为 test；
2. 倒数第二个交互作为 validation；
3. 其余作为 train。

用途：

- 与 HM4SR、RoTE、SASRec、BERT4Rec 等方法更公平对比；
- 作为 robustness check。

### 4.3 Ranking 设置

主实验建议使用 **full ranking**：

- 对所有候选 item 排序；
- 不做 sampled negative evaluation；
- 与 HM4SR 的 whole item set ranking 保持一致。

如果数据集太大，例如 H&M full dataset，可以使用：

- 100 negative samples + all positives，参考 MTSTRec；
- 但需要在论文中明确说明，不能和 full ranking 指标直接混比。

---

## 5. 评估指标

### 5.1 主指标

推荐使用：

- Recall@K / HR@K；
- NDCG@K；
- MRR@K。

K 取：

- `K = 5, 10, 20`。

建议主表报告：

- Recall@10；
- NDCG@10；
- MRR@10；
- 可附 Recall@5 / NDCG@5 / Recall@20 / NDCG@20。

### 5.2 显著性检验

- 每个实验运行 5 个 random seeds；
- 报告 mean ± std；
- 对最强 baseline 使用 paired t-test；
- 显著性阈值：`p < 0.05`。

### 5.3 时间感知诊断指标

为了证明不是普通多模态融合带来的提升，需要额外做时间分组评估：

#### 5.3.1 Δt 分组性能

按目标交互与上一次交互的间隔分组：

| Group | 条件 |
|---|---|
| Very Short | Δt < 1 day |
| Short | 1 day ≤ Δt < 7 days |
| Medium | 7 days ≤ Δt < 30 days |
| Long | Δt ≥ 30 days |

报告每组的 NDCG@10 / Recall@10。

预期：

- 长间隔组提升最明显；
- 因为 continuous-time dynamics 能处理 irregular gap。

#### 5.3.2 Item 波动性分组性能

计算每个 item 在时间窗口中的流行度波动：

$$Volatility(v) = Std\left(\{count_v(w_1), count_v(w_2), ..., count_v(w_T)\}\right)$$

按 volatility 分成 low / medium / high 三组。

预期：

- high-volatility items 上，item distribution ODE 带来更明显提升。

#### 5.3.3 Price-sensitive 分组

如果有价格信息：

- 根据 item price variance 或 category price variance 分组；
- 对 price-change / high-price-variance item 进行单独评估。

预期：

- 价格模态 ODE 在价格敏感组提升更明显。

#### 5.3.4 Cold-start / Long-tail 分组

按 item 训练期交互次数分组：

| Group | 条件 |
|---|---|
| Cold item | 5-10 interactions |
| Medium item | 10-50 interactions |
| Popular item | >50 interactions |

预期：

- 多模态特征 + item distribution ODE 对 cold/long-tail item 有帮助。

---

## 6. Baseline 设计

Baseline 需要覆盖五类：经典 SR、time-aware SR、continuous-time SR、多模态 SR、最新 multimodal temporal/SSM/flow 方法。

### 6.1 经典序列推荐 Baselines

| Method | 类型 | 作用 |
|---|---|---|
| GRU4Rec | RNN-based SR | 经典序列模型 |
| Caser | CNN-based SR | 经典序列模型 |
| SASRec | Transformer-based SR | 最常用强 baseline |
| BERT4Rec | Bidirectional Transformer SR | 常用强 baseline |
| CORE | Session/SR baseline | HM4SR/TGODE 中常用对比 |
| LRURec | Linear recurrent unit | 较新的 efficient SR baseline，可选 |

最低必跑：

- GRU4Rec
- SASRec
- BERT4Rec

### 6.2 Time-aware Sequential Recommendation Baselines

| Method | 核心 | 为什么需要 |
|---|---|---|
| TiSASRec | time interval-aware self-attention | 经典时间间隔 baseline |
| MEANTIME | multi-temporal embeddings | 多种时间 embedding baseline |
| FEARec | time/frequency hybrid attention | HM4SR 对比过 |
| TiCoSeRec | time interval augmentation + contrastive learning | 时间增强 SR baseline |
| RoTE-SASRec | coarse-to-fine rotary time embedding | 与多粒度时间编码最相关 |

最低必跑：

- TiSASRec
- FEARec 或 TiCoSeRec
- RoTE-SASRec

### 6.3 Continuous-time / ODE Baselines

| Method | 核心 | 为什么需要 |
|---|---|---|
| GDERec | Graph ODE for continuous-time SR | ODE 推荐基础 baseline |
| GNG-ODE | Graph nested GRU ODE | TGODE 对比方法 |
| TGODE | Time-guided Graph Neural ODE | 最接近 ODE baseline，必须重点对比 |
| CT4Rec | consistency/time-aware SR | 可选 |

最低必跑：

- GDERec
- TGODE

注意：这些方法通常不是多模态。公平对比有两种方式：

1. 使用原始 ID-only 版本；
2. 增加 multimodal-enhanced 版本，例如把多模态 item feature concat 到 item embedding 后输入 GDERec/TGODE。

建议在表中同时报告：

- TGODE；
- TGODE-MM：TGODE + multimodal item embedding。

这样可以证明提升不只是“加了多模态”。

### 6.4 Multi-modal Sequential Recommendation Baselines

| Method | 核心 | 为什么需要 |
|---|---|---|
| UniSRec | text semantic transfer | 常用多模态/文本增强 baseline |
| MISSRec | multimodal interest-aware sequence pretraining | 多模态 SR 强 baseline |
| M3SRec | modality-specific / cross-modal MoE | 与 MoE、多模态专家相关 |
| MMSR | adaptive multimodal fusion | CIKM 2023，多模态融合 baseline |
| TedRec | ID + text semantic fusion in frequency domain | HM4SR 对比 baseline |
| IISAN | multimodal PEFT / adaptation | HM4SR 对比 baseline |
| HM4SR | Hierarchical Time-Aware MoE | 最接近 time-aware multimodal baseline，必须重点对比 |
| MTSTRec | time-aligned shared token fusion | 与跨模态时间对齐最相关 |

最低必跑：

- UniSRec
- MISSRec
- M3SRec 或 MMSR
- HM4SR
- MTSTRec

### 6.5 最新 Multimodal Temporal / SSM / Flow Baselines

如果代码可用，建议加入：

| Method | 核心 | 为什么需要 |
|---|---|---|
| FindRec | Stein-guided entropic flow + multimodal SR | KDD 2025，强调 temporal dynamics 和 information flow |
| M³Rec | Selective State Space + Mixture-of-Modality Experts | 多模态 SR + SSM + modality experts |
| MMM4Rec | State Space Duality for MM-SR | SSM/Mamba 方向强相关 |
| DuAF-MAT | modality imbalance + dynamic user interests | AAAI 2026，动态多模态兴趣 |

这些可以作为 “recent strong baselines”。如果实现成本高，至少在论文中作为 related work 分析；实验中优先跑代码可复现的 1-2 个。

### 6.6 Baseline 最小可行集合

如果时间有限，建议第一轮实验跑以下 12 个：

1. GRU4Rec
2. SASRec
3. BERT4Rec
4. TiSASRec
5. RoTE-SASRec
6. GDERec
7. TGODE
8. TGODE-MM
9. UniSRec
10. MISSRec
11. HM4SR
12. MTSTRec
13. Ours / CTMID

其中 **TGODE-MM** 很关键，用于排除“只是多模态特征更强”的质疑。

---

## 7. 主实验设计

### 7.1 Overall Performance Table

每个 dataset 一张主表，或合并成大表。

建议表格结构：

| Method | Type | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|
| GRU4Rec | ID-only | | | | | |
| SASRec | ID-only | | | | | |
| TiSASRec | Time-aware | | | | | |
| RoTE-SASRec | Time-aware | | | | | |
| GDERec | Continuous-time | | | | | |
| TGODE | Continuous-time | | | | | |
| UniSRec | Multimodal | | | | | |
| HM4SR | Time-aware multimodal | | | | | |
| MTSTRec | Time-aligned multimodal | | | | | |
| CTMID | Ours | | | | | |

### 7.2 预期结果说明

预期 CTMID：

- 相比 ID-only SR 提升明显；
- 相比 time-aware SR 提升明显，因为引入多模态；
- 相比 multimodal SR 提升明显，尤其在时间间隔大/流行度波动大时；
- 相比 TGODE 提升来自多模态异步建模；
- 相比 HM4SR 提升来自连续时间动力学，而非离散时间 routing。

---

## 8. 消融实验设计

消融实验应围绕“每个创新点是否必要”展开。

### 8.1 模块级消融

| Variant | 移除/替换内容 | 验证问题 |
|---|---|---|
| CTMID | 完整模型 | 主模型 |
| w/o MS-ODE | 每个模态独立 ODE 改为共享 ODE | 模态特定动力学是否必要 |
| w/o User-ODE | 移除用户兴趣 ODE，只用 Transformer/attention 更新用户表示 | 用户侧连续兴趣演化是否必要 |
| w/o Item-ODE | 移除 item 分布 ODE，item 表示静态 | item 流行趋势演化是否必要 |
| w/o Dual-ODE | 同时移除 User-ODE 和 Item-ODE | ODE 框架整体贡献 |
| w/o MGT | 移除 multi-granularity time encoding，使用单一 Δt embedding | 多粒度时间编码是否有效 |
| w/o MATW | 移除 modality-adaptive time weights，所有模态共享同一组时间粒度权重 | 模态感知时间尺度是否必要 |
| w/o TAF | 移除 time-aligned fusion，改为 late fusion / concat | 跨模态时间对齐融合是否必要 |
| w/o Tempo-Reg | 移除时间尺度多样性正则 | 是否防止模态 ODE 退化成同速 |
| w/o Contrast | 移除时间一致性对比学习 | 时间平滑/变化约束是否有效 |
| w/o Cross-Pred | 移除跨模态预测辅助任务 | 慢变模态预测快变趋势是否有帮助 |

建议主消融表报告：

- Beauty / Toys / Sports 三个数据集；
- NDCG@10 和 Recall@10。

### 8.2 ODE 替换消融

为了证明不是“任何时间衰减都行”，加入替换模型：

| Variant | 替换方式 | 验证问题 |
|---|---|---|
| ExpDecay | 用固定指数衰减替代 neural ODE | 是否需要可学习非线性动力学 |
| LearnableDecay | 每个模态学习一个 scalar decay rate | 是否 ODE 比简单衰减更强 |
| GRU-D | 用 GRU-D/time-aware GRU 替代 ODE | ODE 是否优于 RNN-style 时间建模 |
| Mamba/SSM | 用 state-space block 替代 ODE | ODE 与 SSM 方向对比 |
| Shared-ODE | 所有模态共用一个 ODE | 是否需要 modality-specific |

这组实验非常重要，因为它能支撑方法选择。

### 8.3 模态级消融

| Variant | 模态设置 | 验证问题 |
|---|---|---|
| ID only | 只用 ID | 基础序列能力 |
| ID + Text | 加 text | 文本贡献 |
| ID + Image | 加 image | 图像贡献 |
| ID + Price | 加 price | 价格贡献 |
| ID + Category | 加 category | 类别贡献 |
| ID + Text + Image | 常见多模态组合 | 与多数 baseline 对齐 |
| All modalities | ID + Text + Image + Price + Category | 完整模态贡献 |
| w/o Text | 移除 text | 文本是否关键 |
| w/o Image | 移除 image | 图像是否关键 |
| w/o Price | 移除 price | 价格是否关键 |
| w/o Category | 移除 category | 类别是否关键 |

### 8.4 模态生命周期假设消融

这组是专门证明你的核心 idea 的：

| Variant | 设置 | 预期 |
|---|---|---|
| Learned-Lifecycle | 完全学习不同模态时间尺度 | 完整模型最好 |
| Fixed-Prior Lifecycle | 使用先验：image 慢、text 中、price 快 | 接近完整模型 |
| Same-Lifecycle | 强制所有模态相同时间尺度 | 明显下降 |
| Swapped-Lifecycle | 反转先验：image 快、price 慢 | 应显著下降 |
| No-Lifecycle-Reg | 不约束不同模态速度 | 可能退化，性能下降 |

指标除了 NDCG，还应报告：

- 各模态 learned speed / Jacobian norm；
- 各模态时间权重分布。

### 8.5 Item Distribution ODE 消融

| Variant | 设置 | 验证问题 |
|---|---|---|
| User-ODE only | 只保留用户兴趣 ODE | item trend 是否必要 |
| Item-ODE only | 只保留 item 分布 ODE | 用户兴趣 ODE 是否必要 |
| Static Item | item 表示不随时间演化 | item trend 贡献 |
| Global Item ODE | item ODE 不分模态 | 模态感知 item trend 是否必要 |
| Modality Item ODE | 完整 item 模态 ODE | 完整方法 |

---

## 9. 深入分析实验

### 9.1 时间间隔分组分析

目的：证明 ODE 对不规则时间间隔有效。

按目标交互的 `Δt = t_target - t_last` 分组：

| Group | 条件 | 预期结果 |
|---|---|---|
| G1 | Δt < 1 day | CTMID 小幅提升 |
| G2 | 1-7 days | CTMID 中等提升 |
| G3 | 7-30 days | CTMID 明显提升 |
| G4 | >30 days | CTMID 最大提升 |

对比：

- SASRec
- TiSASRec
- TGODE
- HM4SR
- CTMID

### 9.2 Item 流行度波动分析

目的：证明 item distribution ODE 有效。

步骤：

1. 将训练时间划分为固定窗口，例如 30 天；
2. 计算每个 item 在每个窗口的 interaction count；
3. 计算 item popularity volatility；
4. 分成 low / medium / high volatility；
5. 分别报告 NDCG@10。

预期：

- high-volatility item 上提升最大；
- w/o Item-ODE 在 high-volatility 组下降明显。

### 9.3 模态演化速度可视化

目的：直接证明模型学到不同模态生命周期。

可视化指标：

1. ODE derivative norm：

$$Speed^{(m)} = \mathbb{E}_{u,t}\left\|\frac{dh_u^{(m)}(t)}{dt}\right\|_2$$

2. Jacobian norm：

$$J^{(m)} = \left\|\frac{\partial f_m}{\partial h^{(m)}}\right\|_F$$

3. learned time-granularity weights：

$$\alpha_y^{(m)}, \alpha_{month}^{(m)}, \alpha_{day}^{(m)}, \alpha_{hour}^{(m)}$$

预期排序：

- image speed 最低；
- category/text 中等；
- price/promotion 最高。

### 9.4 时间粒度权重分析

目的：证明不同模态关注不同时间尺度。

展示 heatmap：

| Modality | Year | Month | Week | Day | Hour |
|---|---:|---:|---:|---:|---:|
| Image | high | high | medium | low | low |
| Text | medium | medium | medium | medium | low |
| Category | high | medium | medium | low | low |
| Price | low | medium | high | high | medium |

Amazon 可能没有小时级，可只展示 year/month/day/week。

### 9.5 Case Study

选择 2-3 个代表用户序列：

1. 长时间未活跃后重新购买；
2. 促销或季节性 item；
3. 视觉偏好稳定但价格/类别偏好变化的用户。

展示：

- 用户历史序列；
- 每个模态兴趣状态随时间变化曲线；
- CTMID vs HM4SR/TGODE 的 top-5 推荐；
- ground truth item；
- 各模态贡献分数。

### 9.6 鲁棒性分析

#### Missing modality

随机 mask 部分 item 的 image/text/price：

- mask ratio = 10%, 30%, 50%；
- 对比 CTMID、HM4SR、MTSTRec、M3SRec。

目的：验证模态特定 ODE 和跨模态融合是否能缓解模态缺失。

#### Noisy modality

对 image/text embedding 添加 Gaussian noise：

- noise std = 0.05, 0.1, 0.2；
- 分析模型鲁棒性。

---

## 10. 参数敏感性实验

### 10.1 ODE solver 步数

| Steps | 说明 |
|---:|---|
| 1 | Euler-like minimal cost |
| 2 | low cost |
| 4 | default |
| 8 | high precision |

报告：

- NDCG@10；
- training time / epoch；
- inference latency。

预期：

- 2-4 步是最佳 trade-off；
- 8 步提升有限但开销明显增加。

### 10.2 ODE solver 类型

| Solver | 说明 |
|---|---|
| Euler | fastest |
| RK4 | stable default |
| Dormand-Prince | adaptive, more accurate but slower |

建议默认：RK4。

### 10.3 Hidden size

测试：

- 32
- 64
- 128
- 256

主实验默认：64 或 128。

### 10.4 Loss 权重

调参范围：

- `λ_tempo ∈ {0.01, 0.05, 0.1, 0.2}`
- `λ_contrast ∈ {0.05, 0.1, 0.2, 0.5}`
- `λ_cross ∈ {0.05, 0.1, 0.2, 0.5}`

报告其中最关键的 `λ_tempo` 和 `λ_contrast`。

### 10.5 Time window size

用于 item distribution graph：

- 7 days
- 14 days
- 30 days
- 60 days

预期：

- Amazon 上 30 days 可能较稳；
- H&M 上 7/14 days 可能更好。

---

## 11. 效率实验

参考 RoTE 和 TGODE，报告：

1. #Params；
2. FLOPs；
3. training time / epoch；
4. inference latency per batch；
5. GPU memory；
6. ODE solver steps 对效率的影响。

建议表格：

| Method | Params | FLOPs | Train Time/Epoch | Inference Latency | GPU Memory | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| SASRec | | | | | | |
| HM4SR | | | | | | |
| TGODE | | | | | | |
| MTSTRec | | | | | | |
| CTMID | | | | | | |

目标：证明 CTMID 虽然比普通 Transformer 慢，但相对 TGODE/HM4SR 仍可接受。

---

## 12. 实验实现细节

### 12.1 统一设置

建议使用 RecBole 或自建 PyTorch 框架。

统一参数：

| Parameter | Value |
|---|---|
| hidden size | 64 / 128 |
| max sequence length | 50 |
| batch size | 256 / 512 / 1024，视 GPU 而定 |
| optimizer | Adam / AdamW |
| learning rate | 1e-3, 5e-4, 1e-4 grid search |
| dropout | 0.1 / 0.2 / 0.3 |
| weight decay | 1e-5 / 1e-4 |
| early stopping | validation NDCG@10, patience=10 |
| seeds | 5 seeds |

### 12.2 公平性原则

1. 所有方法使用相同 train/val/test split；
2. 所有方法使用相同 max sequence length；
3. 多模态 baseline 使用相同离线 image/text feature；
4. hidden size 尽量统一；
5. 每个 baseline 允许在 validation set 上调参；
6. main table 使用每个方法最优 validation 参数。

### 12.3 Baseline 改造原则

对于不支持多模态的 ODE baseline：

- 原始版本保留；
- 增加 MM-enhanced 版本：把 ID/text/image/price/category embedding 加和或 concat 后投影为 item embedding；
- 这样可以公平比较“多模态 + ODE”是否足够，还是必须做 modality-specific ODE。

---

## 13. 推荐论文中的实验结构

最终论文实验部分可以按以下顺序写：

### 5.1 Experimental Setup

- Datasets；
- Baselines；
- Evaluation metrics；
- Implementation details。

### 5.2 Overall Performance

- 主表；
- RQ1 回答。

### 5.3 Ablation Study

- 模块消融；
- ODE 替换消融；
- RQ2 回答。

### 5.4 Analysis of Asynchronous Modality Dynamics

- learned speed/Jacobian；
- time-granularity weights；
- swapped lifecycle prior；
- RQ3 回答。

### 5.5 Performance under Temporal Scenarios

- Δt 分组；
- item volatility 分组；
- cold-start / long-tail；
- RQ4 回答。

### 5.6 Efficiency and Parameter Sensitivity

- solver steps；
- runtime；
- hidden size；
- RQ5 回答。

### 5.7 Case Study

- 展示用户序列和模态兴趣演化。

---

## 14. 最小可执行实验包

如果时间/算力有限，建议先完成以下最小版本：

### Datasets

- Beauty
- Toys
- Sports

### Baselines

- SASRec
- TiSASRec
- RoTE-SASRec
- TGODE
- TGODE-MM
- UniSRec
- HM4SR
- MTSTRec
- CTMID

### Metrics

- Recall@10
- NDCG@10
- MRR@10
- Recall@20
- NDCG@20

### Ablations

- w/o MS-ODE
- w/o User-ODE
- w/o Item-ODE
- w/o MGT
- w/o MATW
- w/o TAF
- Shared-ODE
- ExpDecay

### Analysis

- Δt 分组；
- item volatility 分组；
- learned modality speed 可视化；
- ODE steps 效率分析。

这个最小实验包已经足够支撑一篇完整论文的主要 claim。

---

## 15. 预期实验结论

如果 hypothesis 成立，应观察到：

1. CTMID 在 overall performance 上超过 HM4SR、TGODE-MM、MTSTRec；
2. w/o MS-ODE 明显下降，证明模态特定动力学必要；
3. w/o Item-ODE 在 high-volatility item 上下降明显；
4. Same-Lifecycle / Swapped-Lifecycle 表现差于 Learned-Lifecycle；
5. learned speed 呈现 `price/category > text > image` 或相近趋势；
6. 长时间间隔用户序列上提升最大；
7. ODE solver 2-4 步达到最佳效率-效果平衡。

---

## 16. 最推荐的主实验 claim

最终论文中可以主张：

> The performance gains are not merely due to using multimodal features or adding temporal embeddings. They come from explicitly modeling asynchronous modality-specific continuous-time dynamics, where each modality evolves at its own temporal scale.

中文：

> 模型性能提升并不只是因为使用了多模态特征或时间 embedding，而是因为显式建模了不同模态在连续时间中的异步演化，使每种模态能够按照自己的时间尺度更新用户兴趣和 item 趋势。
