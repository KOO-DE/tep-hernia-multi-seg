# nnU-Net

| Item | Value |
|---|---|
| Upstream | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| Commit | TODO |
| Code license | Apache-2.0 |
| Pretrained weights | None, unless changed for this study. TODO: Confirm. |

## Setup

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git upstream
cd upstream
git checkout TODO_COMMIT
```

```bash
conda create -n nnunet python=TODO -y
conda activate nnunet
pip install -r ../requirements.txt
```

TODO: Add any other install steps, including the PyTorch and CUDA build.

## Changes for this study

TODO: List each changed upstream file and the reason, or write "None". Put the changed files in `modified_files/`, using the same relative paths as upstream.

If this folder contains `modified_files/`, also add the upstream license here as `LICENSE.upstream`.

## Notes

- TODO: Confirm the nnU-Net version (v2) and the configuration used (e.g. 2d).
- TODO: Confirm the trainer and folds used, and whether an ensemble was reported.
- TODO: Note the dataset ID and how the data were converted to the nnU-Net format.

## Train

```bash
TODO: training command
```

## Inference

```bash
TODO: inference command
```

Save predictions as single-channel PNG masks of class indices, as described in [data/README.md](../../data/README.md).
