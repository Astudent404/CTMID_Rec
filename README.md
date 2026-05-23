# CTMID PyTorch

本项目实现 `Continuous-Time Multimodal Interest Dynamics for Sequential Recommendation`，现在使用自建 PyTorch 数据管线、模型、训练与评估代码，**不再依赖 RecBole**。

## 数据集

`data/` 下提供 3 个 Amazon Reviews 2023 原始数据集，每个由 reviews + meta 两个 `.jsonl.gz` 组成：

| 数据集 | reviews | meta | 配置文件 |
|---|---|---|---|
| All_Beauty | `data/All_Beauty.jsonl.gz` | `data/meta_All_Beauty.jsonl.gz` | `configs/all_beauty_ctmid.yaml` |
| Toys_and_Games | `data/Toys_and_Games.jsonl.gz` | `data/meta_Toys_and_Games.jsonl.gz` | `configs/toys_and_games_ctmid.yaml` |
| Home_and_Kitchen | `data/Home_and_Kitchen.jsonl.gz` | `data/meta_Home_and_Kitchen.jsonl.gz` | `configs/home_and_kitchen_ctmid.yaml` |

reviews 字段：`rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase`；
meta 字段：`main_category, title, average_rating, rating_number, features, description, price, images, videos, store, categories, details, parent_asin, bought_together`。

模态映射：

- `ID`：`user_id` 和 `parent_asin`。
- `time`：`timestamp`，毫秒级 Unix 时间戳。
- `text`：商品 `title/store/main_category/features/description/details`（写入 `item_text.jsonl`）。
- `image`：商品图片 URL（写入 `image_urls.jsonl`）。
- `category`：优先 `main_category`，缺失时回退 `categories`。
- `price`：`log1p` + z-score，缺失用 missing mask。
- `rating`：商品 `average_rating/rating_number`，z-score 归一。

注意：当前没有本地图像文件，也没有预提取 text/image embedding。预处理只生成 `item_text.jsonl` / `image_urls.jsonl` 清单；要启用 text/image 模态需后续离线抽取 `text_features.npy` / `image_features.npy`。`All_Beauty` 2023 类目极稀疏（用户数接近交互数），5-core 过滤后规模很小，更适合做冒烟验证；主实验建议用 `Toys_and_Games` 与 `Home_and_Kitchen`。

## 目录结构

```text
ctmid_recbole/
├── ctmid/
│   ├── data/
│   │   ├── amazon_preprocess.py
│   │   ├── dataset.py
│   │   ├── feature_store.py
│   │   └── collate.py
│   ├── model/
│   │   ├── ctmid.py
│   │   ├── feature_encoder.py
│   │   ├── modality_ode.py
│   │   ├── time_encoding.py
│   │   ├── cross_modal_fusion.py
│   │   └── losses.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/
├── configs/
│   ├── all_beauty_ctmid.yaml
│   ├── toys_and_games_ctmid.yaml
│   ├── home_and_kitchen_ctmid.yaml
│   ├── all_beauty_smoke.yaml
│   └── ablations/
├── scripts/
│   ├── preprocess_amazon.py        # 通用预处理 CLI
│   ├── preprocess_all.sh           # 顺序预处理全部数据集
│   ├── preprocess_all_beauty.py    # 兼容旧入口
│   └── smoke_check.py
├── data/
├── run_ctmid.py
├── requirements.txt
└── README.md
```

## 环境安装

建议 Python 3.9 或 3.10。先按服务器 CUDA 版本安装 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
```

`torchdiffeq`、`sentence-transformers`、`transformers`、`Pillow`、`torchvision` 是可选特征扩展依赖；基础 CTMID 代码默认只需要 PyTorch、NumPy、PyYAML、tqdm 等。

## 数据预处理

通用 CLI 适用于全部数据集：

```bash
python scripts/preprocess_amazon.py --config configs/all_beauty_ctmid.yaml
python scripts/preprocess_amazon.py --config configs/toys_and_games_ctmid.yaml
python scripts/preprocess_amazon.py --config configs/home_and_kitchen_ctmid.yaml
```

或一次性预处理全部数据集：

```bash
bash scripts/preprocess_all.sh
```

预处理流程：流式解析 → 字符串 id 整数化 → 删除 `(user,item,timestamp)` 精确重复行
→ 迭代 5-core 过滤 → 重新编号 → 构建用户序列 → 同时生成两套切分协议
→ 流式解析 meta 一次构造 item 特征。JSON 解析用多进程并行（`--num-workers`，默认
`min(32, cpu_count-2)`）。

每个数据集的输出目录（`{ds}` ∈ `all_beauty / toys_and_games / home_and_kitchen`）：

```text
data/processed/{ds}/
├── user2id.json            # 共享：id 映射 / 文本/图片清单 / 统计
├── item2id.json
├── category2id.json
├── item_text.jsonl
├── image_urls.jsonl
├── stats.json
├── chronological/          # 主协议：全局时间 80/10/10
│   ├── item_features.npz
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── leave_one_out/          # 补充协议：末次=test、倒二=valid
    ├── item_features.npz
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

`data.split` 控制生成哪些协议：`both`（默认）/ `chronological` / `leave_one_out`。

只做字段解析和统计检查，不写完整 processed 文件（`--limit` 限制读取行数）：

```bash
python scripts/preprocess_amazon.py --config configs/all_beauty_ctmid.yaml --limit 200000 --dry-run
```

## 文本特征抽取

预处理只生成 `item_text.jsonl` 清单。用 sentence-transformers 把 item 文本离线编码为冻结
embedding（`all-MiniLM-L6-v2`，384 维），输出 `text_features.npy` 到数据集根目录：

```bash
python scripts/extract_text_features.py --config configs/all_beauty_ctmid.yaml
python scripts/extract_text_features.py --config configs/toys_and_games_ctmid.yaml
python scripts/extract_text_features.py --config configs/home_and_kitchen_ctmid.yaml
```

注意：本服务器无法直连 `huggingface.co`，脚本已默认把 `HF_ENDPOINT` 指向 `hf-mirror.com`。
PAD 行与无文本的 item 置零向量。抽取完成后 3 个数据集的配置已默认启用 `text` 模态。

## 本地轻量检查

不会运行训练循环，只构造一个小 synthetic batch，检查模型 forward、loss、full-sort shape 和 ranking metrics：

```bash
python scripts/smoke_check.py --config configs/all_beauty_smoke.yaml
```

## 服务器训练

预处理完成后，在服务器上运行：

```bash
python run_ctmid.py --config configs/all_beauty_ctmid.yaml
```

只构建数据加载器和模型，不训练：

```bash
python run_ctmid.py --config configs/all_beauty_ctmid.yaml --no-train
```

消融实验示例：

```bash
python run_ctmid.py --config configs/all_beauty_ctmid.yaml --config configs/ablations/without_ms_ode.yaml
python run_ctmid.py --config configs/all_beauty_ctmid.yaml --config configs/ablations/without_item_ode.yaml
python run_ctmid.py --config configs/all_beauty_ctmid.yaml --config configs/ablations/without_mgt.yaml
python run_ctmid.py --config configs/all_beauty_ctmid.yaml --config configs/ablations/without_taf.yaml
```

### 训练提速

为缩短 epoch 时间做了三项优化：

- **`train.num_negatives`**：训练用 sampled-softmax 负采样（默认 4096），每个 batch 只对
  "目标 + 负例" 计算 item 表征，而非全量 item。置 `0` 回退 full-softmax。
- **`model.lambda_rank`**：sampled-softmax CE 之外的 pairwise ranking 辅助项，默认 `0.0` 关闭。
  开启时用正样本 logit 对 sampled negatives 做 `softplus(neg - pos)`。
- **`model.last_item_residual_weight`**：最近一次交互 item 与候选 item 的 ID residual 打分权重，
  默认 `0.0` 关闭；训练、single predict 与 full-sort predict 使用同一分支。
- **eval item 表征缓存**：评估时全量 item 表征只算一次而非每个 batch 重算。
- **`train.amp`**：bf16 混合精度（默认 `false`）。CTMID 的 ODE 数值对 bf16 敏感、会导致
  训练发散，且本模型受每批次开销主导、bf16 提速有限，故默认关闭。

配合 `batch_size: 2048`，Toys 上单 epoch 从 ~18min 降到 ~2.5min（训练 ~2min + 评估 ~0.5min），
评估全量排序口径不变。

Toys 下一轮实验使用叠加配置，避免覆盖当前 `checkpoints/toys_and_games_ctmid_6modal_idres0p5`：

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p75.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_rank0p1.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p25.yaml
```

## 当前默认模态

`All_Beauty` 与 `Toys_and_Games` 的配置默认启用 6 模态：

```yaml
data:
  text_feature_path: data/processed/<dataset>/text_features.npy
  image_feature_path: data/processed/<dataset>/image_features.npy
model:
  modalities: [id, category, price, rating, text, image]
train:
  checkpoint_dir: checkpoints/<dataset>_ctmid_6modal
```

`Home_and_Kitchen` 仍默认启用 5 模态（`id, category, price, rating, text`）。待
`data/processed/home_and_kitchen/image_features.npy` 生成后，把
`configs/home_and_kitchen_ctmid.yaml` 的 `image_feature_path` 指向该文件，并将 `image`
加入 `model.modalities`，同时使用独立的 `_6modal` checkpoint 目录，避免覆盖 5 模态结果。

## 评估协议

预处理同时生成两套切分：

- **chronological（主协议）**：对全部过滤后交互的时间戳取 80% / 90% 分位点，目标交互按时间落入 train / valid / test；每个用户除首个交互外的每个交互都作为一个目标样本，输入仅含其时间之前的历史。
- **leave_one_out（补充协议）**：每个用户最后一个交互为 test，倒数第二个为 valid，其余 prefix-to-next 样本为 train；与 HM4SR / RoTE 对齐。

训练默认读取 `chronological/`（config 的 `data.processed_dir` 指向该子目录）。改用 leave-one-out 时，把 `processed_dir` 末段换成 `leave_one_out` 即可。

评估指标：

- Recall@K
- Precision@K
- Hit@K
- MRR@K
- NDCG@K

默认 `topk: [5, 10, 20]`，主指标 `NDCG@10`。

## 本地 8GB 显存注意事项

不要在本地直接运行完整训练。可以做：

- `python -m compileall ctmid scripts run_ctmid.py`
- `python scripts/preprocess_all_beauty.py --limit 1000 --dry-run`
- `python scripts/smoke_check.py --config configs/all_beauty_smoke.yaml`

完整训练和 full ranking 评估建议上传服务器执行。

## 常见问题

### 为什么没有使用 RecBole？

CTMID 需要原始文本、图片 URL、可选视觉 embedding、模态特定 ODE 和自定义评估/分析流程。RecBole 的 atomic 数据接口对这类多模态序列推荐实验限制较多，因此已迁移为纯 PyTorch 实现。

### All Beauty 是否“完美适配”？

它非常适合作为第一版数据集，但不是完全开箱即用：需要预处理，图像只有 URL 没有本地 embedding，price 缺失严重，`categories` 基本为空。代码已针对这些问题做了默认处理。

### 图片 URL 是否等于图像模态？

不是。URL 只是图像来源。要让模型真正使用视觉模态，需要下载图片并提取视觉向量，例如 CLIP/ViT embedding，然后保存为 `image_features.npy`。

### 文本模态当前是否直接训练？

第一版会生成 `item_text.jsonl`。如果没有 `text_features.npy`，模型不会启用 text 模态。建议后续用 `sentence-transformers` 离线提取 item-level 文本 embedding。
