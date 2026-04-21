# Repository Guidelines

## Project Structure & Module Organization

OpenCOOD is a Python/PyTorch cooperative perception framework. Core code lives in `opencood/`.

- `opencood/tools/`: training, inference, and utility entry points.
- `opencood/hypes_yaml/`: experiment configuration files. Naming usually follows `{backbone}_{fusion_strategy}.yaml`, for example `point_pillar_intermediate_fusion.yaml`.
- `opencood/data_utils/`: datasets, augmentors, preprocessors, and postprocessors.
- `opencood/models/`: model definitions and reusable submodules; fusion modules live in `opencood/models/fuse_modules/`.
- `opencood/loss/`: loss implementations.
- `opencood/utils/`: point cloud, box, transform, evaluation, and compiled helper utilities.
- `opencood/visualization/`: dataset and inference visualization helpers.
- `docs/`, `images/`, and `logreplay/`: documentation, static assets, and CARLA replay tooling.

No dedicated test suite is present.

## Build, Test, and Development Commands

Set up the environment:

```bash
conda env create -f environment.yml
conda activate opencood
python setup.py develop
```

Compile box overlap/NMS helpers:

```bash
python opencood/utils/setup.py build_ext --inplace
```

Compile FPV-RCNN dependencies when needed:

```bash
python opencood/pcdet_utils/setup.py build_ext --inplace
```

Train a model:

```bash
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml
```

Run inference:

```bash
python opencood/tools/inference.py --model_dir opencood/logs/<run> --fusion_method intermediate
```

Visualize a data sequence:

```bash
python opencood/visualization/vis_data_sequence.py --color_mode intensity
```

## Coding Style & Naming Conventions

Use Python 3.7 compatible code. Follow the existing style: 4-space indentation, snake_case for files/functions/variables, and PascalCase for classes. Model and loss names are dynamically imported from YAML, so file and class names must match the convention: `point_pillar_intermediate.py` contains `PointPillarIntermediate`; `point_pillar_loss.py` contains `PointPillarLoss`.

Keep configuration-driven behavior in YAML where possible. Avoid hard-coded dataset paths in source code; use `root_dir`, `validate_dir`, and relevant YAML fields.

## Testing Guidelines

There is no configured pytest or CI suite. Validate changes with the smallest affected workflow: build extensions after native-code changes, run visualization after data pipeline changes, run one short training/inference pass after model, loss, postprocessor, or dataset changes. Use small OPV2V/V2XSet subsets when available.

## Commit & Pull Request Guidelines

This directory is not currently a git repository, so commit history conventions cannot be inferred. Use concise, imperative commit messages such as `Fix PointPillar voxel collation` or `Add V2XSet visualization config`.

Pull requests should include a summary, affected YAML/config files, dataset assumptions, commands run, and observed metrics or artifacts. For visualization changes, attach before/after images or saved output paths.

## Configuration & Data Tips

Before training or inference, confirm that YAML paths point to existing data. Check `root_dir`, `validate_dir`, `fusion.core_method`, `model.core_method`, and `loss.core_method` together, because these fields control dynamic loading and expected batch structure.
