# CXR Abnormalities Detection

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#data">Data</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a>
</p>

## About

Detection of abnormalities on chest X-ray images (VinBigData / VinDr-CXR, 14 finding classes). Every image is annotated by three radiologists, so the boxes are merged with Weighted Boxes Fusion before training and the box score is the fraction of radiologists who agree on it. The detector is a torchvision Faster R-CNN v2 (ResNet-50 FPN) initialized from the COCO weights and fine-tuned with k-fold cross-validation. The metric is mAP@0.4, as in the competition.

## Installation

### Requirements
- Python 3.15.2
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/cxr-abnormalities-detection.git
cd cxr-abnormalities-detection

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.15.2
source .venv/bin/activate

# Install dependencies via uv
uv sync

# Install pre-commit
pre-commit install
```

The scripts that download and process the data need extra packages (kagglehub, pandas):

```bash
uv sync --extra download
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

The API key is only needed for online logging: `writer.mode` is `offline` by default.

## Data

Run the scripts in this order. Everything is stored under `data/vinbigdata/`.

```bash
# 1. Download the resized images and train.csv (Kaggle) -> data/vinbigdata/raw/<size>/
uv run python scripts/download_vinbigdata.py --image-size 512

# 2. Merge the radiologists' boxes (WBF), make the multilabel stratified k-fold split
#    -> data/vinbigdata/preprocessed/{annotations.json,params.json,summary.txt}
uv run python scripts/preprocess_vinbigdata.py --size 512 --folds 5 --wbf-iou 0.4
```

`annotations.json` holds one record per image: the fused boxes (normalized to [0, 1], so they do not depend on the image size), the boxes of each radiologist, the soft labels (the fraction of radiologists who marked each class), the `normal` flag and the `fold`. `params.json` stores the hyperparameters of the run and `summary.txt` the class distribution over the folds. Run each script with `--help` for the options.

## How To Use

To run a script (training or inference), use the following command:

```bash
uv run <script> -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `<script>`: Use `train` for training or `inference` for inference
- `CONFIG_NAME`: A config file from `src/lab/configs` (`train` and `inference` by default)
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

Runs are saved to `logs/{project_name}/{run_name}`, with checkpoints in the `checkpoints` subdirectory.

## Credits

The code of this project is based on the [PyTorch project template](https://github.com/Blinorot/pytorch_project_template).
