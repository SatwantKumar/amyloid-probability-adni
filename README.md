# Amyloid Probability (ADNI) - reproducible analysis code (no data included)

This repository contains code to reproduce the analysis pipeline and figures for the accompanying manuscript on **pattern-to-probability reporting of amyloid biomarker discordance** using ADNI data.

## What this repo includes

- A reproducible “evidence-core” pipeline to standardize ADNI tables and build paired PET/CSF/plasma triads.
- Paper-level analysis code for:
  - **Paper B**: Bayesian latent class modeling for PET–CSF–plasma patterns, timing-stratified discordance, and prespecified sensitivity analyses.
- Utility scripts (for example, PubMed citation formatting from E-utilities metadata).

## What this repo does *not* include

- Any **ADNI raw data** (no `data/raw/` downloads).
- Any **derived ADNI data** (no `core/`, `audit/`, `manifests/`, or `outputs/` artifacts).
- Any participant-level information outside what is distributed by ADNI to approved users.

You must obtain ADNI access and comply with the ADNI data use agreement to run the pipeline.

## Quickstart

### 1) Create an environment and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Configure your ADNI file paths

Edit `config/pipeline.yaml` and point each `sources:` entry to your local ADNI download filenames.

### 3) Build the evidence core (canonical tables + triads)

```bash
python scripts/build_all.py --config config/pipeline.yaml
```

This will create local (ignored) folders: `core/`, `audit/`, `manifests/`.

### 4) Build the Paper B analysis pack

Edit `config/paperB_definitions.example.yaml` as needed, then run:

```bash
python scripts/12_build_paperB_pack.py \
  --definitions config/paperB_definitions.example.yaml \
  --out-dir outputs/paperB
```

This will create local (ignored) artifacts under `outputs/paperB/` (figures, tables, audit CSVs).

## Notes on reproducibility

- The default configurations assume **one PET-anchored triad per participant**, selected deterministically.
- The Paper B workflow uses weakly informative priors and fixed RNG seeds (configurable in `config/paperB_definitions.example.yaml`).

## License

MIT license (see `LICENSE`).

