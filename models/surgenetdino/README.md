# SurgeNetDINO

> This repository is being updated. Reproduction instructions and the evaluation code will be finalized before publication.

This model uses a SurgeNetDINO pretrained ViT backbone in the EoMT segmentation framework.

| Item | Value |
|---|---|
| Framework | [tue-mps/eomt](https://github.com/tue-mps/eomt) |
| Commit | `0f1e1ccbc0e105f1a100e885174e1d716ad02990` (TODO: confirm with `git rev-parse HEAD`) |
| Framework license | MIT, see `LICENSE.upstream` |
| Pretrained weights | DINOv2 ViT-L/14 from [rlpddejong/SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO) (`DINOv2_ViTl14_size336_SurgeNetXL.pth`). TODO: Describe how it was converted to the file used in the config. |
| Weights license | CC-BY-NC-SA (non-commercial, share-alike), which also covers derivative models |

## Setup

Run these commands from this folder.

```bash
conda create -n surgenetdino python=TODO -y
conda activate surgenetdino
pip install -r requirements.txt

git clone https://github.com/tue-mps/eomt.git upstream
cd upstream
git checkout 0f1e1ccbc0e105f1a100e885174e1d716ad02990

cp ../main.py .
cp ../training/lightning_module.py ../training/mask_classification_semantic.py ../training/plot_callback.py training/
cp ../datasets/tep_semantic.py datasets/
cp ../models/eomt_do.py models/
cp ../configs/eomt_large_504_real.yaml ../configs/eomt_large_504_synthetic.yaml configs/
```

Run the training and inference commands from `upstream/`.

TODO: Add the PyTorch and CUDA install command if `requirements.txt` does not cover it.

## Changes for this study

| File | Change | Affects results |
|---|---|---|
| `training/lightning_module.py` | Interpolate backbone position embeddings when the checkpoint and model resolutions differ. Allow backbone-only checkpoints by accepting missing EoMT head keys and unexpected pretraining-only keys. | Yes |
| `training/mask_classification_semantic.py` | Add `freeze_strategy` (`full`, `decoder` or `last2`) to choose which backbone layers are trained. | Yes |
| `training/lightning_module.py` | Log validation pixel accuracy and foreground Dice. Skip WandB logging when no logger is set. | Only if used to select checkpoints |
| `main.py` | Import `training.plot_callback`, disable the learning rate monitor and allow running without a logger. | No |

`models/eomt.py` differs from upstream only by a comment and a line ending, so it is not included here.

Files written for this study:

| File | Purpose |
|---|---|
| `configs/eomt_large_504_real.yaml` | Training config for the model comparison. |
| `configs/eomt_large_504_synthetic.yaml` | Training config for the synthetic data experiment. |
| `datasets/tep_semantic.py` | Loads RGB PNG frames and index masks. Can apply weaker augmentation to synthetic frames, identified by file name. |
| `models/eomt_do.py` | EoMT with one dropout layer, used only for uncertainty estimation. Adapted from `models/eomt.py`. |
| `training/plot_callback.py` | Saves training curves. |
| TODO | Weight conversion, inference and uncertainty scripts. |

## Data

```text
data/TEP/
├── train/             # Real frames only
├── train_synthetic/   # Real and synthetic frames
├── tune/
├── int_val/
└── ext_val/           # Each split has images/ and masks/
```

Synthetic frames were generated from training patients only. They are not distributed. In `train_synthetic`, they are identified by the `_75` or `_85` file name suffix and use weaker augmentation.

| Experiment | Config | Training split |
|---|---|---|
| Model comparison (real data only) | `configs/eomt_large_504_real.yaml` | `train` |
| Synthetic data experiment | `configs/eomt_large_504_synthetic.yaml` | `train_synthetic` |

The synthetic data experiment was performed on SurgeNetDINO, the best-performing model in the comparison.

TODO: Describe `ext_val`.

## Train

| Setting | Value |
|---|---|
| Backbone | `vit_large_patch14_dinov2`, fully fine-tuned (`freeze_strategy` default `full`) |
| Input size | 504×504 |
| Queries | 50 |
| Optimizer | Learning rate 4e-5, backbone multiplier 0.1, layer-wise decay 0.8, weight decay 0.01 |
| Training | Up to 100 epochs, batch size 4, early stopping with patience 30 |
| Augmentation | Color jitter and scale jitter (0.5 to 2.0) on real frames |
| Checkpoint selection | Best `metrics/val_dice_fg` on `tune` |

```bash
python main.py fit -c configs/eomt_large_504_real.yaml
```

## Uncertainty

The comparison results use `models.eomt.EoMT` as set in the config. `models/eomt_do.py` adds one dropout layer (p = 0.1) after block index 13 of the 24-block ViT-L backbone and keeps it active at inference.

TODO: Describe the number of samples, the uncertainty measure and the script.

## Inference

```bash
TODO: inference command
```

This script is used to generate predicted masks. Its printed metrics are not the reported results. All reported metrics were computed with `evaluation/evaluation.py`.
