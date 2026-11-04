# MedSAM

> This repository is being updated. Reproduction instructions and the evaluation code will be finalized before publication.

| Item | Value |
|---|---|
| Upstream | [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) |
| Commit | `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488` (TODO: confirm with `git rev-parse HEAD`) |
| Code license | Apache-2.0, see `LICENSE.upstream` |
| Pretrained weights | MedSAM checkpoint from the link in the upstream README. TODO: Confirm the file used (e.g. `medsam_vit_b`) and its license. |

## Setup

Run these commands from this folder.

```bash
conda create -n medsam python=TODO -y
conda activate medsam
pip install -r requirements.txt

git clone https://github.com/bowang-lab/MedSAM.git upstream
cd upstream
git checkout d71e8a1a99ad751840a22a7fa3ecfb4166fb1488
pip install -e .

cp ../segment_anything/build_sam.py segment_anything/
cp ../segment_anything/modeling/multi_class_mask_decoder.py segment_anything/modeling/
cd ..
```

Run the training and inference commands from this folder.

TODO: Add the PyTorch and CUDA install command if `requirements.txt` does not cover it.

## Changes for this study

| File | Change | Affects results |
|---|---|---|
| `segment_anything/build_sam.py` | Accept checkpoints nested under `model` or `state_dict` and remove the `module.` prefix. Load with `strict=False` and print any missing or unexpected keys. Move the checkpoint download into a helper function. | Only if keys are missing or unexpected |
| `segment_anything/modeling/multi_class_mask_decoder.py` | New file adapted from SAM's `mask_decoder.py`. Replace the binary output with one class token and one output MLP per class, and predict all classes in one forward pass. Initialize the decoder from the pretrained weights, reusing the pretrained mask tokens and MLPs cyclically when there are more classes than tokens. | Yes |

The image encoder, prompt encoder and transformer are unchanged. The mask decoder is replaced as described above.

Scripts in this folder:

| File | Purpose |
|---|---|
| `train_multi_gpus.py` | Multi-class training. Uses the decoder above with weighted CE, generalized Dice and focal Tversky losses, class-balanced image sampling, EMA, a warmup-cosine learning rate and early stopping. |
| `dataset.py` | Loads RGB PNG frames and index masks, resized to 1024×1024. Training applies foreground-focused ROI crop, horizontal flip and photometric augmentation. Validation applies no augmentation. |
| TODO | Inference script. |

## Data

```text
data/TEP/
├── train/
│   ├── images/
│   └── masks/
└── tune/
    ├── images/
    └── masks/
```

An image and its mask share a file name. The first run writes `dataset_stats_num_classes5.json` to each data folder. These files are not part of this repository.

## Prompts

The reported runs use `--prompt_mode full`, which passes a fixed box covering the whole image to the prompt encoder. No box is derived from the ground truth. TODO: Confirm against the saved checkpoint.

The decoder outputs five logits (background and four anatomical structures), so no per-class box or mask merging is needed. TODO: Confirm `num_classes` against the saved checkpoint.

## Train

```bash
python train_multi_gpus.py \
  --train_root data/TEP/train \
  --val_root data/TEP/tune \
  --checkpoint <medsam_vit_b.pth> \
  --work_dir output \
  --prompt_mode full \
  --use_amp
```

TODO: Replace with the options saved in the checkpoint under `args`.

The script selects the best checkpoint by validation macro foreground Dice, evaluated with the EMA weights. `medsam_model_best_macro_dice.pth` stores the raw weights under `model` and the EMA weights under `ema_shadow`. TODO: State which weights were used for the reported results.

## Inference

```bash
TODO: inference command
```

Save predictions as single-channel PNG masks of class indices, as described in [data/README.md](../../data/README.md).
