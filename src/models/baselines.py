from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

def build_model(name, X, seed):
    preprocess = build_preprocessor(X)
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise NotImplementedError(f"Model '{name}' is declared for a later implementation phase")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
