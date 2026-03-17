import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os


def train():
    # ── Load Data ──────────────────────────────────────────
    df = pd.read_csv("data/engineered.csv")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Dataset shape: {X.shape}")
    print(f"Churn distribution:\n{y.value_counts()}")

    # ── Train/Test Split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Handle Class Imbalance with SMOTE ──────────────────
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    # ── Models to try ──────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
        ),
    }

    results = {}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        # Cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train_sm, y_train_sm, cv=cv, scoring="roc_auc"
        )

        # Train and evaluate
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"\n{name}")
        print(f"  CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test AUC:  {auc:.4f}")
        print(f"  Report:\n{classification_report(y_test, y_pred)}")

    # ── Pick Best Model ────────────────────────────────────
    best_name = max(results, key=lambda x: results[x]["auc"])
    best_model = results[best_name]["model"]
    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
    print("=" * 60)

    # ── Save Best Model ────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    print(f"\n✅ Best model saved to models/best_model.pkl")

    return best_model, X_test, y_test


if __name__ == "__main__":
    train()
