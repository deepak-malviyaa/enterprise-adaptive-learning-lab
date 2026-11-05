# Implementation status

## Implemented
- Colab bootstrap notebook
- Configuration validation
- Device detection with CPU fallback
- Synthetic binary quickstart
- Leakage-safe stratified train/validation/test split
- Logistic regression and random-forest pipelines
- PR-AUC, ROC-AUC, precision, recall, F1, and Brier score
- Persisted predictions, metrics, configuration, environment details, and Markdown report
- Unit tests for configuration, data contract, split disjointness, and smoke execution

## Declared but not yet implemented
- Dataset-specific cleaning for ULB, IBM Telco, and AI4I
- XGBoost/LightGBM/CatBoost adapters
- MLP, residual MLP, SwiGLU, FT-Transformer, LSTM, TCN, GNN, MoE, CNN
- Gymnasium environments, DQN, PPO, and GRPO
- MLflow integration and Streamlit dashboard

This distinction prevents placeholder interfaces from being advertised as working features.
