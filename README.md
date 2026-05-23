# RA-CTMID

Residual-Anchored Continuous-Time Multimodal Interest Dynamics for sequential recommendation.

This repository contains a native PyTorch implementation of CTMID and its current strongest variant,
**RA-CTMID**. The project no longer depends on RecBole. It includes preprocessing, multimodal feature
loading, model training, full-sort evaluation, ablation configs, and diagnostic scripts.

## Core Idea

Multimodal sequential recommendation usually fuses item ID, category, price, rating, text, and image
features into a single representation. This improves semantic coverage, but it can also weaken the
strong collaborative filtering signal carried by item IDs. In this project, the main research question is:

```text
How can a continuous-time multimodal recommender preserve semantic and temporal benefits
without diluting ID-based collaborative signals?
```

RA-CTMID answers this by keeping the continuous-time multimodal backbone and adding two residual
collaborative anchors:

- **Global ID Interest Residual**: anchors the score to the user's long-term collaborative preference.
- **Local Last-Item Transition Residual**: anchors the score to short-term item-to-item transitions.

The final score is:

```text
score(u, i, t)
  = s_fuse(u, i, t)
  + alpha * s_id(u, i, t)
  + beta  * s_last(u, i)
  + b_i
```

where `s_fuse` is the continuous-time multimodal backbone score, `s_id` is the long-term ID residual,
`s_last` is the last-item transition residual, and `b_i` is the item output bias.

## Why The Modules Help

Each module is designed to correct a different source of ranking error.

| Module | Signal captured | Why it can improve recommendation |
|---|---|---|
| Multimodal backbone | ID, category, price, rating, text, image | Adds semantic and attribute evidence beyond sparse item co-occurrence. |
| Continuous-time dynamics | Real timestamp intervals | Models interest drift and irregular gaps between interactions. |
| Multi-granularity time encoding | year, month, week, day, interval | Captures short-term recency, periodicity, and seasonal effects. |
| Modality-specific ODE | Different temporal behavior per modality | Prevents stable modalities and fast-changing modalities from sharing one time dynamic. |
| Time-aligned fusion | Modal states at the target time | Fuses user/item states in the same temporal context. |
| ID interest residual | Long-term collaborative preference | Restores ID co-occurrence signals that may be weakened by multimodal fusion. |
| Last-item residual | Short-term item transition | Captures session-like continuation, substitutes, complements, and same-category browsing. |
| Output bias | Global item popularity | Provides an item-level popularity prior. |

The important design choice is not simply adding more components. The model separates three
complementary ranking signals:

```text
multimodal temporal compatibility
+ long-term collaborative preference
+ short-term local transition
```

## Current Finding

On `Toys_and_Games`, the best current configuration is:

```yaml
model:
  modalities: [id, category, price, rating, text, image]
  id_residual_weight: 0.5
  last_item_residual_weight: 0.5
  lambda_rank: 0.0
```

Validation result:

| Experiment | Best Epoch | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| 6modal + ID residual 0.5 + last-item residual 0.5 | 2 | 0.0171388738 | 0.0101219559 | 0.0079798321 | 0.0239735906 | 0.0118399515 |

Key observations:

| Experiment | NDCG@10 | Interpretation |
|---|---:|---|
| 6modal baseline | 0.0071135995 | Multimodal backbone helps but is not enough. |
| 6modal + ID residual 0.5 | 0.0092682372 | Long-term collaborative anchoring gives a large gain. |
| 6modal + ID residual 1.0 | 0.0094952427 | Stronger ID anchoring still helps, with diminishing return. |
| 6modal + ID residual 0.5 + last residual 0.25 | 0.0097031097 | Local transition anchoring adds further gain. |
| 6modal + ID residual 0.5 + last residual 0.5 | 0.0101219559 | Current best setting. |
| 6modal + ID residual 0.5 + rank loss 0.1 | 0.0091019640 | Pairwise rank loss did not help. |
| 6modal + ID residual 0.5 + rank loss 0.3 | 0.0090932433 | Stronger rank loss still did not help. |

This suggests that the bottleneck is less about adding a more complex sequence encoder or loss, and
more about preserving collaborative structure after multimodal fusion.

## Repository Structure

```text
.
├── ctmid/
│   ├── data/                  # preprocessing output readers and feature store
│   ├── model/                 # CTMID and RA-CTMID model components
│   ├── training/              # trainer, evaluator, ranking metrics
│   └── utils/
├── configs/
│   ├── ablations/             # backbone ablation configs
│   ├── experiments/           # residual, sweep, and RA experiment configs
│   ├── all_beauty_ctmid.yaml
│   ├── toys_and_games_ctmid.yaml
│   └── home_and_kitchen_ctmid.yaml
├── scripts/
│   ├── preprocess_amazon.py
│   ├── extract_text_features.py
│   ├── extract_image_features.py
│   ├── evaluate_ra_ctmid_diagnostics.py
│   ├── run_ra_ctmid_all.sh
│   └── smoke_check.py
├── run_ctmid.py
├── requirements.txt
└── README.md
```

The repository intentionally excludes raw data, processed data, extracted features, checkpoints, and
logs. See `.gitignore`.

## Data

The code targets Amazon Reviews 2023 style review/meta files. Put raw files under `data/` locally:

| Dataset | Reviews file | Meta file | Base config |
|---|---|---|---|
| All Beauty | `data/All_Beauty.jsonl.gz` | `data/meta_All_Beauty.jsonl.gz` | `configs/all_beauty_ctmid.yaml` |
| Toys and Games | `data/Toys_and_Games.jsonl.gz` | `data/meta_Toys_and_Games.jsonl.gz` | `configs/toys_and_games_ctmid.yaml` |
| Home and Kitchen | `data/Home_and_Kitchen.jsonl.gz` | `data/meta_Home_and_Kitchen.jsonl.gz` | `configs/home_and_kitchen_ctmid.yaml` |

Mapped modalities:

- `id`: `user_id` and `parent_asin`
- `time`: millisecond Unix `timestamp`
- `category`: `main_category`, with fallback to `categories`
- `price`: `log1p` plus z-score, with missing mask
- `rating`: item `average_rating` and `rating_number`, z-score normalized
- `text`: item title, store, category, features, description, and details
- `image`: item image URLs, later encoded into frozen image embeddings

Raw `data/` is not tracked by git.

## Installation

Use Python 3.9 or 3.10. Install the PyTorch wheel matching the server CUDA version first, then:

```bash
pip install -r requirements.txt
```

`torchdiffeq`, `sentence-transformers`, `transformers`, `Pillow`, and `torchvision` are optional for
advanced ODE solvers and offline text/image feature extraction.

## Preprocessing

Run preprocessing for one dataset:

```bash
python scripts/preprocess_amazon.py --config configs/toys_and_games_ctmid.yaml
```

Or preprocess all configured datasets:

```bash
bash scripts/preprocess_all.sh
```

The preprocessing pipeline performs streaming parsing, ID remapping, exact duplicate removal,
iterative 5-core filtering, user sequence construction, chronological splitting, leave-one-out
splitting, and item feature construction.

Output layout:

```text
data/processed/<dataset>/
├── user2id.json
├── item2id.json
├── category2id.json
├── item_text.jsonl
├── image_urls.jsonl
├── stats.json
├── chronological/
│   ├── item_features.npz
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── leave_one_out/
    ├── item_features.npz
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

For a lightweight parsing check:

```bash
python scripts/preprocess_amazon.py --config configs/toys_and_games_ctmid.yaml --limit 200000 --dry-run
```

## Feature Extraction

Preprocessing only creates text and image URL manifests. To enable frozen text features:

```bash
python scripts/extract_text_features.py --config configs/toys_and_games_ctmid.yaml
python scripts/extract_text_features.py --config configs/home_and_kitchen_ctmid.yaml
```

To enable frozen image features:

```bash
python scripts/extract_image_features.py --config configs/toys_and_games_ctmid.yaml
python scripts/extract_image_features.py --config configs/home_and_kitchen_ctmid.yaml
```

The expected files are:

```text
data/processed/<dataset>/text_features.npy
data/processed/<dataset>/image_features.npy
```

## Training

Run the base Toys configuration:

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml
```

Build data loaders and model without training:

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --no-train
```

Run the current RA-CTMID best setting on Toys:

```bash
python run_ctmid.py \
  --config configs/toys_and_games_ctmid.yaml \
  --config configs/experiments/toys_6modal_idres0p5_last0p5.yaml
```

Run the current RA-CTMID setting on Home and Kitchen:

```bash
python run_ctmid.py \
  --config configs/home_and_kitchen_ctmid.yaml \
  --config configs/experiments/home_6modal_idres0p5_last0p5.yaml
```

Configuration files are merged from left to right, so experiment configs override the base config.

## Key Experiment Configs

Residual ablations:

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_nores.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_last0p5.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p5.yaml
```

Beta sweep with `alpha = 0.5`:

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p1.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p25.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p5.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last0p75.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_6modal_idres0p5_last1p0.yaml
```

Backbone ablations under the RA setting:

```bash
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_without_ms_ode.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_without_user_ode.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_without_item_ode.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_without_mgt.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_without_taf.yaml
python run_ctmid.py --config configs/toys_and_games_ctmid.yaml --config configs/experiments/toys_ra_exp_decay.yaml
```

## Evaluation Protocol

The main protocol is chronological full-sort evaluation:

- interactions are split by global timestamp into train/valid/test
- each target is predicted from its historical prefix only
- history items are masked during ranking
- metrics are computed with `topk: [5, 10, 20]`
- the main validation metric is `NDCG@10`

Supported metrics:

- Recall@K
- Precision@K
- Hit@K
- MRR@K
- NDCG@K

The preprocessing code also generates a leave-one-out protocol for comparison with sequence
recommendation baselines.

## Diagnostics

Use diagnostics to explain why residual anchoring works rather than only reporting scores:

```bash
python scripts/evaluate_ra_ctmid_diagnostics.py --config configs/toys_and_games_ctmid.yaml
```

Recommended analyses:

- group by `delta_t = target_time - last_interaction_time`
- group by user history length
- group by item popularity
- group by whether target and last item share category
- compare score components: `s_fuse`, `alpha * s_id`, `beta * s_last`, and `output_bias`

The expected mechanism is:

- last-item residual helps more for very short and short time gaps
- ID residual helps medium/long history users by preserving collaborative preference
- the continuous-time multimodal backbone remains important for semantic matching and longer gaps

## Development Checks

Compile check:

```bash
python -m compileall ctmid scripts run_ctmid.py
```

Synthetic smoke check:

```bash
python scripts/smoke_check.py --config configs/all_beauty_smoke.yaml
```

## Notes

- Do not commit `data/`, `checkpoints/`, `features/`, `saved/`, or `log/`.
- `All_Beauty` is useful for smoke tests but is sparse after 5-core filtering.
- `Toys_and_Games` and `Home_and_Kitchen` are the main datasets for the RA-CTMID paper story.
- Image URLs are not image features. Extract `image_features.npy` before enabling the image modality.
