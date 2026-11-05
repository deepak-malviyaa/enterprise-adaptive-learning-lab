from sklearn.model_selection import train_test_split

def make_splits(frame, target, config):
    X = frame.drop(columns=[target]); y = frame[target]
    split = config["split"]
    test_f = float(split["test_fraction"]); val_f = float(split["validation_fraction"])
    stratify = y if split["strategy"] == "stratified" else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_f, random_state=config["seed"], stratify=stratify)
    relative_val = val_f / (1 - test_f)
    stratify_tmp = y_tmp if split["strategy"] == "stratified" else None
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=relative_val, random_state=config["seed"], stratify=stratify_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test
