# Model Zoo

## Pre-training LaViLa

### 
<details><summary> Train a baseline dual-encoder with ViT-B </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 \
    scripts/main_lavila_pretrain.py \
    --root /new-pool/Datasets/Ego4d/v1/videos_288px_15sec/ \
    --root-val datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

<details><summary> Train an LM-augmented dual-encoder (LaViLa) with ViT-B </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 \
    scripts/main_lavila_pretrain.py \
    --root /new-pool/Datasets/Ego4d/v1/videos_288px_15sec/ \
    --root-val datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --train-metadata datasets/Ego4D/ego4d_train.rephraser.no_punkt_top3.pkl \
    --train-metadata-aux datasets/Ego4D/ego4d_train.narrator_63690737.return_10.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

|  Corpus  | LLM-aug. | Corpus size | Backbone | per-gpu<br>batch-size | GPU×hour^ | EK-100 MIR<br>avg. mAP | EK-100 MIR<br>avg. nDCG |                                checkpoint                               | md5sum |
| :------: | :------: | :---------: | :------: | :----------------: | :-------: | :--------------------: | :---------------------: | :---------------------------------------------------------------------: | :----: |
|  Ego4D   |    no    |   4.0M      |  ViT-B   |       256          |  ~130    |       27.5/28.4        |       29.1/29.5         | [best Epoch](https://drive.google.com/file/d/1l-kaIHSoXSOtyEEzhE4CQVFRbEH1P2F7/view?usp=drive_link) | fc3b7f |
|  Ego4D   |    yes   |    35M      |  ViT-B   |       256          |   ~260    |       31.1/32.9        |       31.9/32.7         | [best Epoch](https://drive.google.com/file/d/1CiHEIdFSUut6mIweqObrsdbMMCPasSLQ/view?usp=sharing) | 91a90b |
|  Ego4D   |    yes   |    35M      |  ViT-L   |       112          |   ~680    |       36.4/37.6        |       35.1/35.3         | [best Epoch](https://drive.google.com/file/d/1NvlW5KQnPEr435EeRGmVf9BN6uv-s9BY/view?usp=sharing) | f377f6 |



^ Hardware configuration: 8x NVIDIA A5000 (24GB) GPUs + 2x Intel Xeon Gold 5220(R) 24-Core CPU @ 2.20GHz (96 threads in total).

## Fine-tuning the video-language dual-encoder on down-stream tasks

### EK-100 Multi-Instance Retrieval (MIR)

<details><summary> Finetune a pretrained dual-encoder on EK-100 MIR </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>


| LLM-aug. | Backbone | V->T mAP | T->V mAP | avg mAP | V->T nDCG | T->V nDCG | avg nDCG |                               checkpoint                                | md5sum |
| :------: | :------: | :------: | :------: | :-----: | :-------: | :-------: | :------: | :---------------------------------------------------------------------: | :----: |
|   yes    |   ViT-B  |   55.7   |   48.2   |  52.0   |   67.8    |   65.3    |   66.5   | [best epoch](https://drive.google.com/file/d/1cVLsfjSHI0_7DeLKMjdHrSX-UKLmZKhE/view?usp=sharing) | e099c0 |
|   yes    |   ViT-L  |   57.9   |   51.1   |  54.5   |   70.4    |   67.6    |   69.0   | [best epoch](https://drive.google.com/file/d/1Fnw7lb7Gw0MZf41R9kXf64mUanvIGIw8/view?usp=sharing) | f82079 |


### EK-100 Action Recognition (CLS)


<details><summary> Finetune a pretrained dual-encoder on EK-100 CLS </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_cls.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

<details><summary> Evaluate the model after fine-tuning </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_cls.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --resume ${PATH_TO}/avion_finetune_cls_lavila_vitb_best.pt \  # additional to the training script
    --evaluate                                                    # additional to the training script
```

</details>

| LLM-aug. | Backbone | Verb Top1 | Noun Top1 | Action Top1 |                                checkpoint                               | md5sum |
| :------: | :------: | :-------: | :-------: | :---------: | :---------------------------------------------------------------------: | :----: |
|   no     |   ViT-B  |   67.9    |   57.6    |    47.3     | [best epoch](https://drive.google.com/file/d/1-5v_c3UGnuSU_wIsgKmsbBAhsbY_Y8mP/view?usp=sharing) | b40f3e |
|   yes    |   ViT-B  |   70.0    |   59.4    |    49.5     | [best epoch](https://drive.google.com/file/d/1YgxYjmpSxI26wdnKeeT0YakEIAFP5HjF/view?usp=sharing) | 6c3c5e |
|   yes    |   ViT-L  |   73.0    |   65.4    |    54.4     | [best epoch](https://drive.google.com/file/d/1HSL6AQox5FGwp_yZBHgECkj_gvifA3EN/view?usp=sharing) | 1871f4 |


## Pre-training and Fine-tuning VideoMAE

###
<details><summary> Train a VideoMAE on Kinetics with ViT-B </summary>

```bash
mkdir experiments/videomae_pretrain_vitb_lion/
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=4 scripts/main_videomae_pretrain.py \
    --model VIDEOMAE_VITB16 \
    --use-flash-attn-at-encoder --use-flash-attn-at-decoder \
    --batch-size 64 --channel-last \
    --fused-decode-crop --use-multi-epochs-loader --optimizer lion \
    -j 8 \
    --output-dir experiments/videomae_pretrain_vitb_lion 2>&1 | tee experiments/videomae_pretrain_vitb_lion/log.txt
```

</details>

<details><summary> Finetune a pretrained VideoMAE model on Kinetics-400 </summary>

```bash
# training
mkdir experiments/videomae_finetune_vitb_lion_e800/
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_videomae_finetune.py \
    --use-flash-attn --channel-last \
    --finetune experiments/videomae_pretrain_vitb_lion/checkpoint_00800.pt \
    -j 8 \
    --output-dir experiments/videomae_finetune_vitb_lion_e800/ 2>&1 | tee experiments/videomae_finetune_vitb_lion_e800/log.txt
```

```bash
# testing
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_videomae_finetune.py \
    --use-flash-attn --channel-last \
    --finetune experiments/videomae_pretrain_vitb_lion/checkpoint_00800.pt \
    -j 8 \
    --output-dir experiments/videomae_finetune_vitb_lion_e800/ \
    --evaluate \
    --resume experiments/videomae_finetune_vitb_lion_e800/checkpoint_best.pt 2>&1 | tee experiments/videomae_finetune_vitb_lion_e800/eval_log.txt
```
</details>

|           | Backbone |   Top-1   |   Top-5   |                                checkpoint                                 | md5sum |
| :-------: | :------: | :-------: | :-------: | :-----------------------------------------------------------------------: | :----: |
| pre-train |   ViT-B  |   67.9    |   57.6    | [700-th epoch](https://drive.google.com/file/d/1Uuc1Wjc41MSYNibux7L4zs0XePZw5IxM/view?usp=sharing) | 2bbcaf |
| fine-tune |   ViT-B  |   80.0    |   94.5    |   [best epoch](https://drive.google.com/file/d/1XnC38Qz2195fLjurpRYCbioMuA48ZVFu/view?usp=sharing) | 5cd5c5 |
