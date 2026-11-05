# Experiment guide

## Mandatory order

1. Environment check.
2. Dataset card and audit.
3. Leakage-safe split tests.
4. Logistic regression and one tree baseline.
5. Generated baseline report.
6. MLP, residual MLP, and parameter-matched SwiGLU.
7. Sequence, graph, MoE, or RL only after core acceptance.

Never tune on test data. Resampling occurs only inside the training partition. Validation data selects thresholds, calibration, and hyperparameters. Keep test prevalence natural.
