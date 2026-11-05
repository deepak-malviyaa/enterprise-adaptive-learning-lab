def validate_frame(frame, target):
    if frame.empty:
        raise ValueError("Dataset is empty")
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' is missing")
    if frame[target].isna().any():
        raise ValueError("Target contains missing values")
    if frame[target].nunique() != 2:
        raise ValueError("Starter classification runner requires a binary target")
    return {"rows": len(frame), "columns": len(frame.columns), "positive_rate": float(frame[target].mean())}
