# MM-TS: Multi-Modal Temperature and Margin Schedules for Contrastive Learning with Long-Tail Data

<p align="center">
  <a href="https://openaccess.thecvf.com/content/WACV2026/papers/Sheludzko_MM-TS_Multi-Modal_Temperature_and_Margin_Schedules_for_Contrastive_Learning_with_WACV_2026_paper.pdf">
    <img src="https://img.shields.io/badge/Paper-WACV%202026-blue" />
  </a>
  <a href="https://github.com/SergShel/MM-TS">
    <img src="https://img.shields.io/badge/Code-GitHub-black" />
  </a>
</p>

> **MM-TS: Multi-Modal Temperature and Margin Schedules for Contrastive Learning with Long-Tail Data**  
> Siarhei Sheludzko¹, Dhimitrios Duka², Bernt Schiele², Hilde Kuehne³˒⁴, Anna Kukleva²  
> ¹University of Bonn &nbsp; ²MPI for Informatics, SIC &nbsp; ³Tuebingen AI Center / University of Tuebingen &nbsp; ⁴MIT-IBM Watson AI Lab  
> **WACV 2026**

---

## Abstract

Contrastive learning has become a fundamental approach in both uni-modal and multi-modal frameworks, pulling positive pairs closer while pushing negatives apart. While the temperature parameter controls the strength of these forces, most approaches simply fix it during training or treat it as a constant hyperparameter.

We propose **MM-TS** (Multi-Modal Temperature and Margin Schedules), extending the concept of uni-modal temperature scheduling to multi-modal contrastive learning. MM-TS dynamically adjusts the temperature in the contrastive loss during training, modulating the attraction and repulsion forces in the multi-modal setting. Additionally, recognizing that standard multi-modal datasets often follow imbalanced, long-tail distributions, we adapt the temperature based on the local distribution of each training sample — samples from dense clusters are assigned a higher temperature to better preserve their semantic structure.

Furthermore, we demonstrate that temperature scheduling can be effectively integrated within a max-margin framework, thereby unifying the two predominant approaches in multi-modal contrastive learning: InfoNCE loss and max-margin objective.

We evaluate on four image- and video-language datasets — **Flickr30K, MSCOCO, EPIC-KITCHENS-100, and YouCook2** — and show that MM-TS improves performance and achieves new state-of-the-art results.

---

## Method Overview

MM-TS combines two components:

1. **Cosine Temperature Schedule** — the base temperature follows a cosine schedule across training iterations, allowing the model to alternate between instance discrimination (low τ) and group-wise discrimination (high τ).

2. **Individual Cluster Shifts (ICS)** — text annotations are embedded and clustered (K-Means) prior to training. Each sample's temperature is shifted based on its cluster size: frequent/head concepts get a higher temperature (promoting semantic grouping), while rare/tail concepts get a lower temperature (enforcing instance separation).

The final per-sample temperature is:

```
τ_i = τ_base(t) + sh(c_i)
```

where `τ_base(t)` follows a cosine schedule and `sh(c_i)` is the cluster-based shift for sample `i`.

MM-TS is also extended to the **max-margin loss** by replacing the fixed margin with the modulated temperature, enabling dynamic margin scheduling for frameworks such as AVION on EPIC-KITCHENS-100.

---

## Results

### EPIC-KITCHENS-100 (Multi-Instance Retrieval)

| Method | Backbone | mAP V→T | mAP T→V | mAP Avg. | nDCG V→T | nDCG T→V | nDCG Avg. |
|---|---|---|---|---|---|---|---|
| MME | TBN | 43.0 | 34.0 | 38.5 | 50.1 | 46.9 | 48.5 |
| JPoSE | TBN | 49.9 | 38.1 | 44.1 | 55.5 | 51.6 | 53.5 |
| EgoVLP | TSF-B | 49.9 | 40.5 | 45.0 | 60.9 | 57.9 | 59.4 |
| LaViLa | TSF-B | 55.2 | 45.7 | 50.5 | 66.5 | 63.4 | 65.0 |
| AVION | ViT-B | 55.7 | 48.2 | 52.0 | 67.8 | 65.3 | 66.5 |
| **AVION + MM-TS (Ours)** | ViT-B | **58.8** | **48.9** | **53.9** | **68.9** | **65.8** | **67.3** |

### YouCook2 (Text-to-Video Retrieval)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| UniVL | 28.9 | 57.6 | 70.0 |
| MELTR | 33.7 | 63.1 | 74.8 |
| VLM | 27.1 | 56.9 | 69.4 |
| VAST | 50.4 | 74.3 | 80.8 |
| **VAST + MM-TS (Ours)** | **53.0** | **77.1** | **84.5** |

### Zero-Shot Retrieval on Flickr30K / MSCOCO (Pretrained on CC3M)

| Method | Flickr30K IR@1 | Flickr30K TR@1 | MSCOCO IR@1 | MSCOCO TR@1 |
|---|---|---|---|---|
| CLIP (RN50) | 40.9 | 50.9 | 21.3 | 26.9 |
| **CLIP + MM-TS (Ours)** | **41.5** | **54.3** | **21.2** | **28.4** |

---

## Code

This repository contains the official implementation of MM-TS integrated with the **AVION** framework for the EPIC-KITCHENS-100 Multi-Instance Retrieval task.

> **Note:** We conducted experiments with additional contrastive frameworks (CLIP on CC3M, VAST on YouCook2). Code for those will be released in future updates.

### Repository Structure

```
MM-TS/
└── avion/          # MM-TS integrated with the AVION framework (EK-100 MIR)
```

For installation, dataset setup, training, and evaluation instructions, please refer to the [`avion/`](avion/) directory.

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Sheludzko_2026_WACV,
    author    = {Sheludzko, Siarhei and Duka, Dhimitrios and Schiele, Bernt and Kuehne, Hilde and Kukleva, Anna},
    title     = {MM-TS: Multi-Modal Temperature and Margin Schedules for Contrastive Learning with Long-Tail Data},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7376-7386}
}
```

---

## Acknowledgements

Our AVION-based implementation builds on the [AVION](https://github.com/zhaoyue-zephyrus/AVION) codebase. We thank the authors for releasing their code.

---

## License

This project is released under the MIT License.
