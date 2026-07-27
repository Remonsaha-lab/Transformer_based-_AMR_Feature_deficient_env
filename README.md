# EZ-SP / Superpoint Transformer — DIODE Installation and Project Guide

This guide documents the DIODE-only installation and execution workflow for
this repository: prerequisites, environment setup, DIODE preparation, training,
evaluation, visualization, active learning, outputs, and project architecture.

The implementation contains three related systems:

- **Superpoint Transformer (SPT)** for semantic segmentation.
- **SuperCluster** for panoptic segmentation.
- **EZ-SP (Easy Superpoints)**, which learns a GPU-friendly superpoint
  partition and then performs semantic segmentation on that partition.

Run the commands below from:

```text
/home/remon/EZ_SP_CFDVri/superpoint_transformer
```

Replace this path if the repository is located elsewhere.

## 1. Project workflow

The complete point-cloud workflow is:

```text
raw point cloud
    → dataset reader and preprocessing
    → point features, neighborhoods, graphs, and hierarchical superpoints
    → SPT / SuperCluster model
    → semantic or panoptic prediction
    → metrics, checkpoints, logs, and visualizations
```

SPT predicts semantic classes over a superpoint hierarchy. SuperCluster adds
instance-level panoptic predictions by clustering a superpoint graph during
inference. EZ-SP uses two training stages: a point CNN learns embeddings and a
GPU graph partitioner builds superpoints; a second model then performs semantic
segmentation using those learned partitions.

## 2. Requirements

The upstream project was tested on Linux, approximately 64 GB RAM, and NVIDIA
GPUs such as GTX 1080 Ti 11 GB, V100 32 GB, and A40 48 GB. A CUDA GPU is
strongly recommended.

Required tools:

- Linux, preferably Ubuntu 20.04 or newer
- NVIDIA driver compatible with CUDA 11.8 or 12.1
- CUDA toolkit 11.8 or 12.1, including `nvcc`
- Conda/Miniconda
- Git, GCC, and G++

Verify the machine:

```bash
nvidia-smi
nvcc --version
git --version
gcc --version
g++ --version
conda --version
```

For optional TorchSparse support on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libsparsehash-dev
```

## 3. Get the code

```bash
git clone <repository-url> EZ_SP_CFDVri
cd EZ_SP_CFDVri/superpoint_transformer
```

If the checkout already exists:

```bash
cd /home/remon/EZ_SP_CFDVri/superpoint_transformer
```

The main entry points are `src/train.py` and `src/eval.py`. They use
`pyrootutils` to find the project root and add it to `PYTHONPATH`, so the
project does not need to be installed as a Python package.

## 4. Installation

The canonical installer is `install.sh`. It creates a Conda environment named
`spt`, detects CUDA with `nvcc`, installs PyTorch 2.2.0 and PyTorch Geometric,
installs the packages in the project, and compiles FRNN. It supports CUDA
11.8 and 12.1.

### Standard environment

```bash
cd /home/remon/EZ_SP_CFDVri/superpoint_transformer
chmod +x install.sh
./install.sh
conda activate spt
```

### Environment with TorchSparse

TorchSparse is used by the EZ-SP sparse CNN. Install it with:

```bash
cd /home/remon/EZ_SP_CFDVri/superpoint_transformer
./install.sh with_torchsparse
conda activate spt
```

The optional path clones and compiles TorchSparse and installs sparsehash. Use
it for EZ-SP experiments that require sparse 3D convolutions.

### Installed dependency groups

The installer covers PyTorch 2.2.0, PyTorch Geometric 2.3.0, Lightning,
Hydra/OmegaConf, NumPy/SciPy/pandas/scikit-learn, h5py, Open3D, PLY and
geometry packages, `pgeof`, `pycut-pursuit`, `pygrid-graph`, `torch-ransac3d`,
`torch-graph-components`, visualization/logging tools, and the compiled FRNN
extension. The maintained package list is also in
[`requirements.txt`](superpoint_transformer/requirements.txt).

The previous installation note described a Python 3.10 virtual environment.
That is not the repository’s canonical path: the checked-in `install.sh` uses
Conda and Python 3.8. Follow `install.sh` unless you deliberately maintain a
separate, compatible environment.

## 5. Verify the environment

```bash
cd /home/remon/EZ_SP_CFDVri/superpoint_transformer
conda activate spt

python -c "import torch, torch_geometric, pgeof, frnn; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('PyG:', torch_geometric.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If TorchSparse was installed:

```bash
python -c "import torchsparse; print('TorchSparse import: OK')"
```

Run configuration tests before a long training job:

```bash
pytest -q tests/test_configs.py
```

Run all tests when the required data and optional dependencies are available:

```bash
pytest -q
```

## 6. DIODE dataset setup

Download the DIODE dataset manually. The project does not automate downloads
for licensing/compliance reasons. Keep the original DIODE RGB, depth, and mask
files in the DIODE split directories, then convert them to point clouds with
`scripts/load_diode.py`.

Convert DIODE RGB-D frames to PLY point clouds:

```bash
python scripts/load_diode.py \
  --diode_root data/diode/val \
  --out_dir data/diode/raw_ply/val
```

The DIODE data module preprocesses the point clouds into
`data/diode/processed/` on first use. Processed data is grouped by a
preprocessing hash, allowing multiple preprocessing variants to coexist.
For quick experiments, the data module also supports the `mini` option.

Expected high-level layout:

```text
superpoint_transformer/
└── data/
    └── diode/
        ├── raw_ply_data/  # train/test PLY files
        ├── raw_ply/       # validation PLY files
        └── processed/
```

### External data and log directories

To keep large files outside the repository, create
`configs/local/default.yaml`:

```yaml
# @package paths
data_dir: /path/to/your/data/
log_dir: /path/to/your/logs/
```

Without this file, the defaults are `data/` and `logs/` inside the project.

## 7. Hydra configuration

Hydra composes a main config, an experiment preset, and command-line
overrides. The main files are:

- `configs/train.yaml` — training and optional post-training test evaluation.
- `configs/eval.yaml` — evaluation of a required checkpoint.

Important config groups:

```text
configs/
├── datamodule/       dataset readers, preprocessing, and loaders
├── model/            semantic, panoptic, and partition model definitions
├── experiment/       ready-to-run dataset/model combinations
├── trainer/          CPU, GPU, DDP, and memory-specific settings
├── callbacks/        checkpointing and monitoring
├── logger/           W&B, CSV, TensorBoard, MLflow, and other loggers
├── paths/            data, log, output, and working-directory paths
├── debug/            limit, overfit, and profiler settings
└── hydra/            runtime output-directory settings
```

General command form:

```bash
python src/train.py experiment=<task>/diode_ezsp key=value
```

Useful DIODE examples:

```bash
python src/train.py experiment=semantic/diode_11g +datamodule.mini=True
python src/train.py experiment=semantic/diode_11g trainer.max_epochs=50
python src/train.py experiment=semantic/diode_11g logger=tensorboard
python src/train.py experiment=semantic/diode_11g debug=limit
python src/train.py experiment=semantic/diode_11g ckpt_path=/path/to/last.ckpt
```

## 8. Training

### DIODE semantic segmentation

```bash
# DIODE baseline, configured for an 11 GB GPU
python src/train.py experiment=semantic/diode_11g
```

The DIODE baseline uses the SPT semantic model. Adjust `xy_tiling`, batch
size, or `+datamodule.mini=True` for limited GPU memory.

### DIODE EZ-SP two-stage training

**Stage 1: train the partition model.** The point CNN learns an embedding and
the GPU graph partitioner uses it to construct hierarchical superpoints:

```bash
python src/train.py experiment=partition/diode_ezsp
```

The checkpoint is usually under:

```text
logs/train/runs/<run_name>/checkpoints/last.ckpt
```

**Stage 2: train the semantic model** using the Stage 1 checkpoint:

```bash
python src/train.py \
  experiment=semantic/diode_ezsp \
  datamodule.pretrained_cnn_ckpt_path=/absolute/path/to/partition/last.ckpt
```

Stage 2 needs `pretrained_cnn_ckpt_path`; the semantic and partition models
are separate checkpoints.

## 9. Evaluation

Evaluation requires a checkpoint and the matching experiment configuration:

```bash
# DIODE semantic SPT
python src/eval.py experiment=semantic/diode_11g \
  ckpt_path=/absolute/path/to/checkpoint.ckpt
```

DIODE EZ-SP evaluation also needs the Stage 1 checkpoint:

```bash
python src/eval.py experiment=semantic/diode_ezsp \
  ckpt_path=/absolute/path/to/semantic_checkpoint.ckpt \
  datamodule.pretrained_cnn_ckpt_path=/absolute/path/to/partition_checkpoint.ckpt
```

## 10. Visualization and utility scripts

Inspect a script’s arguments first:

```bash
python scripts/visualize_stage1_outputs.py --help
python scripts/visualize_specific_scene.py --help
python scripts/check_validation_superpoints.py --help
python scripts/visualize_downsampling.py --help
```

Stage 1 output visualization defaults to `outputs/stage1_vis`:

```bash
python scripts/visualize_stage1_outputs.py \
  --file_path /path/to/processed_scene.h5 \
  --out_dir outputs/stage1_vis
```

Other repository utilities include:

```bash
python scripts/load_diode.py --help
python scripts/generate_mask2former_labels.py --help
python scripts/check_validation_superpoints.py --help
python scripts/visualize_specific_scene.py --help
python scripts/visualize_downsampling.py --help
python check_duplicates.py --help
```

Some utilities are dataset-specific and may require configuration in the
source file rather than command-line arguments.

## 11. Active learning

`stage1_active_learning.py` selects scans using partition purity scores
(`purity_miou`). `stage2_active_learning.py` selects scans using semantic
scores (`semantic_miou`). Both support an initial random subset and a later
score-based expansion.

```bash
python scripts/stage1_active_learning.py --help
python scripts/stage2_active_learning.py --help
```

Example Stage 1 flow:

```bash
python scripts/stage1_active_learning.py init \
  --all-ids /path/to/all_scan_ids.txt \
  --fraction 0.10 --seed 0 \
  --output active_learning/stage1_round_10.txt

python scripts/stage1_active_learning.py next \
  --current-ids active_learning/stage1_round_10.txt \
  --scores active_learning/scores/stage1_val_purity_epoch_0049.csv \
  --target-fraction 0.20 \
  --output active_learning/stage1_round_20.txt \
  --added-output active_learning/stage1_added_10_to_20.txt
```

Stage 2 uses the same pattern with `stage2_active_learning.py`; its default
score column is `semantic_miou`.

## 12. Project architecture

```text
superpoint_transformer/
├── configs/                  Hydra configuration graph
├── data/                     raw and generated dataset data
├── docs/                     dataset, logging, visualization, and data docs
├── scripts/                  preprocessing, diagnostics, visualization, AL
├── src/
│   ├── data/                 Data, NAG, CSR, cluster, and instance structures
│   ├── datamodules/          PyTorch Lightning data modules
│   ├── datasets/             DIODE reader and point-cloud dataset utilities
│   ├── dependencies/         compiled FRNN and optional TorchSparse sources
│   ├── loader/               data-loading helpers
│   ├── loss/                 semantic, panoptic, partition, and weighted losses
│   ├── metrics/              semantic, panoptic, mAP, and weighted metrics
│   ├── models/               semantic and panoptic Lightning modules
│   │   └── components/       SPT backbone and MLP components
│   ├── nn/                   attention, pooling, sparse, transformer, fusion
│   ├── optim/                optimizers and learning-rate schedulers
│   ├── transforms/           preprocessing, graph, geometry, sampling, device
│   ├── utils/                Hydra, logging, graph, checkpoint, output helpers
│   ├── visualization/        point-cloud visualization functions
│   ├── train.py              training entry point
│   └── eval.py               evaluation entry point
├── tests/                    configuration, train, eval, and sweep tests
├── install.sh                Conda/CUDA installation script
└── requirements.txt          dependency pins
```

### Runtime data flow

1. A dataset class reads raw files and creates the project point-cloud data
   representation.
2. Pre-transforms compute neighborhoods, graph connectivity, features, and
   partition hierarchy, then cache them in `processed/`.
3. DataLoader transforms apply sampling and augmentation.
4. On-device transforms prepare GPU tensors and run expensive GPU operations.
5. The model consumes point/superpoint features and graph connectivity.
6. Lightning computes losses and metrics, saves checkpoints, and supports
   training, testing, and prediction.

### Model components

- `src/models/semantic.py` implements semantic segmentation.
- `src/models/panoptic.py` implements SuperCluster panoptic segmentation.
- `src/models/components/spt.py` contains the Superpoint Transformer backbone.
- `src/nn/` contains attention, transformer, pooling/unpooling, sparse,
  normalization, and fusion layers.
- `src/loss/` and `src/metrics/` contain task-specific objectives and metrics.
- EZ-SP partition models are configured under `configs/model/partition/` and
  `configs/experiment/partition/`.

### Configuration flow

```text
experiment preset
    ↓ overrides
datamodule + model + trainer + logger + callbacks + paths
    ↓
src/train.py or src/eval.py
```

This allows dataset, model, GPU, logger, debug, and checkpoint settings to be
changed from the command line without editing Python code.

## 13. Logs, checkpoints, and outputs

Training creates Hydra run directories with the resolved configuration,
metrics, and Lightning checkpoints. A typical checkpoint location is:

```text
logs/train/runs/<run_name>/checkpoints/
├── last.ckpt
└── <best-checkpoint>.ckpt
```

The exact location is printed by Hydra and depends on
`configs/hydra/default.yaml` and any local path override. In general:

- `data/diode/processed/` contains generated DIODE files.
- `logs/` contains training logs and checkpoints.
- `outputs/` contains generated visualizations and utility outputs.
- W&B is enabled by many experiment presets; use `wandb login` when needed.

To disable online logging for a run:

```bash
python src/train.py experiment=semantic/diode_11g logger=null
```

## 14. Troubleshooting

### CUDA is unavailable

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Fix the driver/toolkit first if `nvidia-smi` or `nvcc` fails. If only PyTorch
fails, verify that the activated environment contains the CUDA-compatible wheel
selected by `install.sh`.

### CUDA out-of-memory

Use an `_11g` experiment, run with `+datamodule.mini=True`, reduce the batch
size, or select a smaller model under `configs/model/`.

### FRNN or TorchSparse compilation fails

Verify `nvcc`, CUDA headers, GCC/G++, and (for TorchSparse) sparsehash. Inspect
the first compiler error in the build output; the final pip error is often only
a summary.

### Dataset files are missing

Check `data/diode/`, the selected experiment name, and
`configs/local/default.yaml`. The configured `data_dir` must contain the
DIODE folder.

### EZ-SP cannot load its partition model

Pass an existing absolute Stage 1 checkpoint:

```bash
datamodule.pretrained_cnn_ckpt_path=/absolute/path/to/partition.ckpt
```

The Stage 1 partition checkpoint and Stage 2 semantic checkpoint are different
files.

## 15. Quick-start checklist

```bash
cd /home/remon/EZ_SP_CFDVri/superpoint_transformer
./install.sh
conda activate spt

python -c "import torch, torch_geometric, pgeof, frnn; print(torch.cuda.is_available())"

# Place the prepared DIODE data under data/diode/
python src/train.py experiment=semantic/diode_11g +datamodule.mini=True

# Evaluate a completed run
python src/eval.py experiment=semantic/diode_11g \
  ckpt_path=/absolute/path/to/checkpoint.ckpt
```

For additional background and exact dataset/logging/visualization details,
see [`README.md`](superpoint_transformer/README.md),
[`docs/datasets.md`](superpoint_transformer/docs/datasets.md),
[`docs/logging.md`](superpoint_transformer/docs/logging.md), and
[`docs/visualization.md`](superpoint_transformer/docs/visualization.md).
