from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def train_bdt_strategy(df, feature_cols, strategy_name, seed=0):
    df = df.copy()

    X = df[feature_cols]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    bdt = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )

    bdt.fit(X_train, y_train)

    y_pred = bdt.predict(X_test)
    y_score = bdt.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    print("\n" + "="*60)
    print(strategy_name)
    print("="*60)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(classification_report(y_test, y_pred, target_names=["DY", "H"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "name": strategy_name,
        "model": bdt,
        "accuracy": acc,
        "auc": auc,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_score": y_score,
    }