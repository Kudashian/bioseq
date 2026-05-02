# Week 8 — MLOps: Experiment Tracking & Reproducibility

## Objective
Instrument the Week 4 CNN with production MLOps practices using 
Weights & Biases.

## What Was Built
- W&B tracking: per-epoch train loss, per-fold accuracy, run summary
- Config-driven training: all hyperparameters live in config.yaml
- Model versioning: best fold weights saved as W&B artifact
- Reproducible runs: seeded torch, numpy, and sklearn splits

## Results
- Mean CV Accuracy: 96.19% (consistent with Week 4 baseline of 97%)
- Best fold: Fold 1 (100%)
- Config: lr=0.001, epochs=50, hidden_size=96, dropout=0.5

## Key Learnings
- W&B config is the single source of truth — code never hardcodes hyperparams
- Artifacts link model weights to the exact run that produced them
- Reproducibility requires seeding weights, shuffling, and data splits
- Summary = outcome, Log = process, Config = input, Artifact = deliverable

## Stack
PyTorch · Weights & Biases · scikit-learn · YAML

## Model Comparison (running total)
| Model | Mean CV Accuracy |
|-------|-----------------|
| CNN (BatchNorm + Dropout) | ~97% |
| DNABERT (fine-tuned) | ~78% |
| Custom LSTM | ~54% |