"""
modelling.py - Workflow CI dengan MLflow Project

Script ini digunakan oleh GitHub Actions untuk melakukan re-training model
secara otomatis ketika trigger workflow berjalan.
Dataset yang digunakan adalah dataset hasil preprocessing, bukan dataset raw.
"""

from pathlib import Path
import json
import os
import pandas as pd
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "breast_cancer_preprocessing"
TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"
TARGET_COL = "target"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"File train tidak ditemukan: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


def main():
    os.environ.pop("MLFLOW_RUN_ID", None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Workflow CI - Breast Cancer Classification")

    # Basic requirement: modelling.py memakai autolog.
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="ci_random_forest_autolog") as run:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Tambahan artifact agar hasil CI mudah diperiksa.
        metrics_path = OUTPUT_DIR / "model_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        report_path = OUTPUT_DIR / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        cm_path = OUTPUT_DIR / "confusion_matrix.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)

        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="evaluation_outputs")

        latest_run_path = OUTPUT_DIR / "latest_run_id.txt"
        latest_run_path.write_text(run.info.run_id, encoding="utf-8")

        print(f"MLflow run_id: {run.info.run_id}")
        print("Training selesai. Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
