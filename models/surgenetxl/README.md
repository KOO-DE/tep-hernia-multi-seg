# SurgeNetXL

> This repository is being updated. Reproduction instructions and the evaluation code will be finalized before publication.
 
| Item | Value |
|---|---|
| Upstream | [TimJaspers0801/SurgeNet](https://github.com/TimJaspers0801/SurgeNet) |
| Commit | `54831cfacf3a6cac0bbf5119f72b9cb9c31c3e82` (TODO: confirm with `git rev-parse HEAD`) |
| Code license | MIT, see `LICENSE.upstream`. `metaformer.py` also keeps its Apache-2.0 file header. |
| Model | `MetaFormerFPN`: CAFormer-S18 encoder and FPN decoder |
| Pretrained weights | SurgeNetXL teacher backbone (`SurgeNetXL_checkpoint_epoch0050_teacher.pth`). TODO: Confirm with `pretrained_ckpt` in the saved checkpoint. |
| Weights license | CC-BY-NC-SA (non-commercial, share-alike), which also covers derivative models |

## Setup

Run these commands from this folder.

```bash
conda create -n surgenetxl python=TODO -y
conda activate surgenetxl
pip install -r requirements.txt
```

`metaformer.py` in this folder is the modified upstream file, and the scripts import it directly. No upstream clone is needed.

TODO: Add the PyTorch and CUDA install command if `requirements.txt` does not cover it.

## Changes for this study

| File | Change | Affects results |
|---|---|---|
| `metaformer.py` | Load pretrained weights from a local file as well as a URL. Accept checkpoints nested under `state_dict`, `model` or `teacher`, and remove the `module.` and `backbone.` prefixes. Print the number of missing and unexpected keys. | No, if the same weights are loaded |
| `metaformer.py` | Add a `verbose` argument, raise an error for an unsupported `pretrained` value, and clean up comments and formatting. | No |

The encoder and FPN decoder architectures are unchanged.

Scripts in this folder:

| File | Purpose |
|---|---|
| `inference.py` | Runs the model, resizes predictions to the original frame size and saves them as single-channel PNG files of class indices. |
| `train.py` | Trains on one GPU or with DDP. Saves `best_dice.pth`, `best_val_loss.pth`, `best_pixel_acc.pth` and `latest.pth`. |

## Train

| Setting | Script default |
|---|---|
| Fine-tuning | `--tuning_mode full` |
| Optimizer | AdamW (encoder learning rate 1e-5, decoder learning rate 1e-4, weight decay 1e-4) |
| Scheduler | 3 warmup epochs, then cosine decay |
| Training | 100 epochs, batch size 4 |
| Augmentation | Horizontal and vertical flips, 90° rotations, brightness and contrast |
| Loss | Cross-entropy, or 0.5 cross-entropy + 0.5 Dice with `--use_dice` |

TODO: Replace the defaults with the values saved in the checkpoint under `args`.

```bash
python train.py \
  --train_img_dir data/TEP/train/images \
  --train_mask_dir data/TEP/train/masks \
  --val_img_dir data/TEP/tune/images \
  --val_mask_dir data/TEP/tune/masks \
  --save_dir output \
  --pretrained_ckpt <SurgeNetXL or a local weight file> \
  --image_size TODO
```

TODO: Add `--use_dice` if it was used. With DDP and `--tuning_mode full`, add `--find_unused_parameters`.

## Inference

```bash
python inference.py \
  --img_dir data/TEP/int_val/images \
  --mask_dir data/TEP/int_val/masks \
  --checkpoint <trained_checkpoint.pth> \
  --save_dir predictions \
  --image_size TODO \
  --save_pred_masks
```

`--image_size` must match the input size used for training. Predictions are saved to `predictions/pred_masks/`.

This script is used to generate predicted masks. Its printed metrics are not the reported results. All reported metrics were computed with `evaluation/evaluation.py`.
