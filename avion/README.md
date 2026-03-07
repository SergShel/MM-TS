# MM-TS for AVION (EK-100 Multi-Instance Retrieval)

This repo contains **MM-TS experiments built on top of [AVION](https://github.com/zhaoyue-zephyrus/avion)**. It provides training + evaluation code for the **AVION backbone** on **EPIC-Kitchens-100 Multi-Instance Retrieval (EK-100 MIR)**.

**What’s new here?**  
MM-TS modifies **only the loss + data handling**. The **model architecture and AVION pretraining are unchanged**.

---

## Table of Contents

- [Setup](#setup)
  - [Environment](#1-environment)
  - [Datasets](#2-datasets)
  - [Pretrained checkpoint](#3-download-pretrained-checkpoint)
- [What changed vs. AVION](#what-changed-vs-avion)
- [Running experiments](#running-experiments)
  - [Generate class distributions (one-time)](#generate-class-distributions-one-time)
  - [Fine-tune with MM-TS](#fine-tune-with-mm-ts)
  - [Loss variants](#loss-variants)
- [MM-TS arguments](#mm-ts-arguments)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Setup

### 1) Environment

Follow AVION’s installation instructions: **[docs/INSTALL.md](docs/INSTALL.md)**.  

### 2) Datasets

See **[datasets/README.md](datasets/README.md)** for downloading and preparing **EPIC-Kitchens-100 / EK-100**.

### 3) Download pretrained checkpoint

```bash
bash scripts/download_checkpoints.sh
```

> **Note:** MM-TS fine-tuned checkpoints will be available soon.

---

## What changed vs. AVION

All additions are at the **loss and data level** (temperature scheduling + class-frequency-based shifts).

### Added utilities

| Path | What it does |
|------|--------------|
| `mmts_utils/temperature.py` | Cosine-oscillating base temperature (`compute_tau_base`) |
| `mmts_utils/shift.py` | Per-sample shift from class frequency (`compute_cluster_based_shift`) |
| `mmts_utils/generate_ek100_distributions.py` | Builds class-frequency CSVs from EK100 annotations |

### Precomputed distributions

| Path | Contents |
|------|----------|
| `data/distributions/ek100_{verb,noun,verb_noun}_freq.csv` | Class-frequency distributions used for per-sample shifts |

---

## Running experiments

### Generate class distributions (one-time)

These are already provided under `data/distributions/`, but you can regenerate them from the EK100 train CSV:

```bash
python mmts_utils/generate_ek100_distributions.py \
    --csv datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --output-dir data/distributions/ \
    --distributions verb noun verb_noun
```

### Fine-tune with MM-TS

```bash
EXP_PATH=experiments/mmts_clip_mir_vitb
mkdir -p $EXP_PATH

PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=4 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/video_320p_15sec \
    --train-metadata datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
    --val-metadata datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --relevancy-path datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --video-chunk-length 15 \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 128 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/avion_pretrain_lavila_vitb_best.pt \
    --loss-type mmts_max_margin \
    --use-distribution noun \
    --mmts-alpha 0.2 \
    --mmts-num-oscillations 5 \
    --mmts-shift-min 0.17 \
    --mmts-shift-max 0.30 \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

---

## Loss variants

Switch `--loss-type` to run different objectives:

| `--loss-type` | Description |
|---------------|-------------|
| `clip` | Standard CLIP InfoNCE (baseline) |
| `max_margin` | Max-margin ranking loss (baseline) |
| `mmts_clip` | MM-TS: per-sample temperature on InfoNCE |
| `mmts_max_margin` | MM-TS: per-sample adaptive margin |

---

## MM-TS arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--loss-type` | `max_margin` | `clip`, `max_margin`, `mmts_clip`, `mmts_max_margin` |
| `--use-distribution` | `noun` | Which class-frequency distribution to use: `noun`, `verb`, `verb_noun` |
| `--mmts-alpha` | `0.2` | Amplitude of cosine temperature oscillation |
| `--mmts-num-oscillations` | `5` | Number of full cosine periods across training |
| `--mmts-shift-min` | `0.17` | Minimum per-sample shift (rare classes) |
| `--mmts-shift-max` | `0.30` | Maximum per-sample shift (frequent classes) |
| `--mmts-noun-dist-path` | `data/distributions/ek100_noun_freq.csv` | Noun frequency CSV |
| `--mmts-verb-dist-path` | `data/distributions/ek100_verb_freq.csv` | Verb frequency CSV |
| `--mmts-verb-noun-dist-path` | `data/distributions/ek100_verb_noun_freq.csv` | Verb–noun frequency CSV |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This project is built on top of **[AVION](https://github.com/zhaoyue-zephyrus/avion)** by **Yue Zhao** and **Philipp Krähenbühl**.