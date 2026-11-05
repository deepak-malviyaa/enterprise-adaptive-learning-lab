# Enterprise Adaptive Learning Laboratory

A portable, configuration-driven research POC for traditional ML, deep learning, and decision learning. It is designed for Google Colab first, with CPU-safe local execution on Windows and macOS.

> Research POC only. Public and synthetic datasets do not establish production effectiveness, regulatory compliance, or real-world savings.

## First implemented vertical slice

The starter skeleton provides a runnable **synthetic binary-classification quickstart** that validates the project plumbing without downloading confidential or third-party data. Dataset adapters for ULB Fraud, IBM Telco Churn, and UCI AI4I are explicit placeholders with data-card templates.

## Colab quickstart

1. Upload `enterprise-adaptive-learning-lab.zip` to Colab.
2. Open `notebooks/00_colab_bootstrap.ipynb`.
3. Run all cells.
4. The notebook extracts the project, installs it, validates the environment, runs tests, and executes:

```bash
python -m src.run --use-case synthetic_binary --mode quickstart --device auto
```

Outputs are written under `outputs/`, including configuration, metrics, predictions, environment details, and a Markdown report.

## CLI

```bash
python -m src.environment_check
python -m src.run --use-case synthetic_binary --mode quickstart
python -m src.run --use-case bfsi_fraud --mode quickstart
python -m src.run --use-case telecom_churn --mode single_use_case
python -m src.run --use-case manufacturing_pm --mode single_use_case
```

The three enterprise dataset commands fail early with clear setup instructions until the approved datasets are placed under `data/raw/`.

## Design rules

- Configuration selects use case, dataset, model family, device, sample cap, training budget, and output location.
- Preprocessing is fitted on train data only.
- Raw data is never committed.
- CPU quickstart is mandatory.
- Every run records seed, inputs, environment, metrics, predictions, and caveats.
- Advanced models and RL are added only after classical baselines and contracts work.
- Synthetic simulator results must remain labelled as simulator-only evidence.

See `docs/architecture.md`, `docs/experiment_guide.md`, and `docs/implementation_status.md`.
