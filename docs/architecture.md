# Architecture

The laboratory separates four concerns:

1. **Representation learning:** how inputs are encoded.
2. **Predictive learning:** what is likely to happen.
3. **Decision learning:** what action should be taken.
4. **Outcome learning:** what happened after the decision.

The runner loads validated YAML, resolves a device, invokes one use-case contract, persists artifacts, and writes a report. Raw data, model code, evaluation, RL environments, and reporting remain separate. Advanced optional dependencies must not be imported by an unrelated CPU quickstart.
