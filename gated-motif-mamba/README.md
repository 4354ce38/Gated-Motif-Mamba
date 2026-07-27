# gated-motif-mamba

Minimal GitHub-ready repo for the `Mamba-2 + input gate + state PQ motif` variant used in the LTC experiments.

This folder keeps:

- the modified `gated_motif_mamba_ssm` core implementation
- a small loader API for base-model + adapter checkpoints
- a TruthfulQA trajectory plotting script
- an S-NIAH evaluation example
- a lightweight ablation recipe example

## What this repo can do

Base model note:

- the base `mamba2-130m` model is not bundled in this repo
- it will be uploaded to Hugging Face later
- until then, put the downloaded model under `assets/models/mamba2-130m/` or pass `--model-dir`

### 1. Load the requested adapter checkpoint

Default adapter location:

`assets/checkpoints/130m_motif2/adapter_latest.pt`

Adapter note:

- the adapter checkpoint is not bundled in this repo
- it will be uploaded to Hugging Face later
- after downloading it, place it at `assets/checkpoints/130m_motif2/adapter_latest.pt` or pass `--adapter`

Demo:

```bash
python scripts/load_adapter_demo.py
```

### 2. Reproduce the TruthfulQA trajectory figure

Default output directory:

`outputs/truthfulqa_internal_explanations`

Run:

```bash
python scripts/plot_truthfulqa_internal_explanations.py
```

This script writes several figures, including:

`truthfulqa_case_pq_trajectories_wrong_base.svg`

### 3. Run an S-NIAH example

```bash
python scripts/run_sniah_mamba2_eval.py --limit 20
```

To evaluate the base model only:

```bash
python scripts/run_sniah_mamba2_eval.py --adapter none
```

### 4. See ablation command examples

```bash
python scripts/ablation_example.py
```

## Local structure

- `assets/models/`
  Reserved location for the base model after it is downloaded separately.
- `assets/checkpoints/`
  Reserved location for adapter checkpoints after they are downloaded separately.
- `assets/data/truthfulqa/`
  Local TruthfulQA snapshot and benchmark CSVs used by the plotting script.
- `assets/data/sniah/`
  Small S-NIAH example dataset bundled in the repo.
- `assets/data/pile_tiny/`
  Tiny tokenized sample for trajectory-analysis demos.
- `gated_motif_mamba/`
  Loader and checkpoint helpers.
- `gated_motif_mamba_ssm/`
  Modified Mamba core.
- `scripts/`
  Reproducible entry points.
- `outputs/`
  Default output location for plots and eval results.

## Install

Install from source in an environment that already has the right CUDA-enabled PyTorch:

```bash
pip install -e . --no-build-isolation
```

If `selective_scan_cuda` is not already available in your environment, this command will build it from `csrc/selective_scan`.

## GitHub Note

The base model is intentionally not included here.
It will be provided later through Hugging Face.

The main adapter checkpoint is also intentionally not included here.
It will be provided later through Hugging Face.
