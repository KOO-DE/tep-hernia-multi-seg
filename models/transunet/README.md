# TransUNet

| Item | Value |
|---|---|
| Upstream | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) |
| Commit | `02ef0010b36eb8328b5e689eadaf613602edf9b8` (TODO: confirm with `git rev-parse HEAD`) |
| Code license | Apache-2.0, see `LICENSE.upstream` |
| Variant | R50-ViT-B_16 |
| Pretrained weights | Google ImageNet-21k `R50+ViT-B_16.npz`. The upstream README says the official download appears to have expired and links a copy in its project folder. |

## Setup

Run these commands from this folder.

```bash
conda create -n transunet python=TODO -y
conda activate transunet
pip install -r requirements.txt

git clone https://github.com/Beckschen/TransUNet.git upstream
cd upstream
git checkout 02ef0010b36eb8328b5e689eadaf613602edf9b8

cp ../train.py ../test.py ../trainer.py ../utils.py .
cp ../datasets/dataset_synapse.py datasets/
cp ../networks/vit_seg_configs.py networks/
```

Run the remaining commands from `upstream/`. Place the weights at:

```text
model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

TODO: Add the PyTorch and CUDA install command if `requirements.txt` does not cover it.

## Data

```text
data/TEP/
├── train/
├── tune/
└── int_val/        # Each split has images/ and masks/
lists/lists_TEP/
├── train.txt
├── tune.txt
└── int_val.txt     # One file name per line, without the extension
```

The list files are not distributed. TODO: Confirm that `int_val` is the test set reported in the paper.

## Changes for this study

| File | Change | Affects results |
|---|---|---|
| `datasets/dataset_synapse.py` | Load RGB PNG frames and index masks instead of `.npz` and `.h5` volumes. Keep three channels through augmentation. | Yes |
| `train.py` | Set 5 classes, batch size 16 and learning rate 0.005. Add loss options and an output path argument. | Yes |
| `trainer.py` | Add validation on the `tune` split, optional weighted CE, boundary and clDice losses, best-checkpoint saving and training curves. | Yes |
| `utils.py` | Run inference on RGB frames and save predictions and overlays as PNG instead of NIfTI. | Yes |
| `test.py` | Use the TEP dataset, R50-ViT-B_16 and a selected checkpoint. Add a checkpoint folder argument. | Yes |
| `networks/vit_seg_configs.py` | Change pretrained weight paths from `../model/` to `model/`. | No |

The model definition in `networks/vit_seg_modeling.py` is unchanged.

## Train

```bash
python train.py \
  --root_path data/TEP \
  --list_dir lists/lists_TEP \
  --output_root output \
  --loss_mode TODO
```

TODO: Add `--class_weights` and any other non-default options used in the paper.

## Inference

```bash
python test.py \
  --volume_path data/TEP \
  --list_dir lists/lists_TEP \
  --snapshot_path <training_output_folder> \
  --checkpoint TODO \
  --is_savenii \
  --test_save_dir predictions
```

`--is_savenii` saves PNG files in this version. Colored predictions are written to `pred/`.

TODO: State how the colored predictions were converted to class indices for `evaluation/evaluation.py`.

The metrics printed by `test.py` follow the upstream code. Report the metrics from `evaluation/evaluation.py`.
