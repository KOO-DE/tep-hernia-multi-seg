# U-Net

> This repository is being updated. Reproduction instructions and the evaluation code will be finalized before publication.

U-Net with a ResNet-50 encoder, built with [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch). The training and test scripts were written for this study.

| Item | Value |
|---|---|
| Encoder | ResNet-50 |
| Pretrained weights | ImageNet, downloaded by `segmentation_models_pytorch` |
| Input size | 512×512 |
| Normalization | ImageNet mean and standard deviation |
| Augmentation | None |
| Loss | Cross-entropy |
| Optimizer | AdamW (learning rate 1e-4, weight decay 1e-4) |
| Scheduler | Cosine annealing to 1e-6 |
| Training | 100 epochs, batch size 8 |

TODO: Confirm these values against the command used for the reported results.

## Setup

Run these commands from this folder.

```bash
conda create -n unet python=TODO -y
conda activate unet
pip install -r requirements.txt
```

TODO: Add the PyTorch and CUDA install command if `requirements.txt` does not cover it.

## Data

```text
data/TEP/
├── train/
├── tune/
└── int_val/     # Each split has images/ and masks/
```

An image and its mask share a file name. Masks are single-channel PNG files of class indices.

TODO: Confirm that `int_val` is the test set reported in the paper.

## Train

```bash
python train.py --data_root data/TEP --save_dir output
```

The script saves `best_dice.pth`, `best_acc.pth` and `best_val_loss.pth`, selected on the `tune` split.

## Inference

```bash
python test.py \
  --data_root data/TEP \
  --checkpoint output/best_dice.pth \
  --output_dir predictions
```

TODO: Confirm which checkpoint was used for the reported results.

The script resizes frames and masks to 512×512 and saves predictions to `predictions/pred_masks/` as color PNG files.

TODO: State how the color predictions were converted to class indices for `evaluation/evaluation.py`, and the resolution used for evaluation.

Report the metrics from `evaluation/evaluation.py`. The metrics written by `test.py` are computed per batch.
