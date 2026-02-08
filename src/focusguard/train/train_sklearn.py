from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from focusguard.data.dataset import load_events_df, summarize_events_df
from focusguard.data.labeling import make_labels
from focusguard.data.quality import check_quality
from focusguard.data.splits import make_splits


def _load_train_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _select_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X.to_numpy(dtype=np.float32)


def _fit_logreg(train_cfg: Dict[str, Any]) -> LogisticRegression:
    mcfg = train_cfg.get("model", {}).get("logreg", {})
    return LogisticRegression(
        C=float(mcfg.get("C", 1.0)),
        class_weight=mcfg.get("class_weight", "balanced"),
        max_iter=int(mcfg.get("max_iter", 200)),
    )


def _eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "labels": labels,
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, lab in enumerate(labels)
        },
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def _has_rows(X: np.ndarray) -> bool:
    return X is not None and X.shape[0] > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()

    train_cfg = _load_train_cfg(Path(args.config))

    # MLflow
    mlflow_cfg = train_cfg.get("mlflow", {})
    mlflow.set_tracking_uri(str(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")))
    mlflow.set_experiment(str(mlflow_cfg.get("experiment", "focusguard")))

    feature_cols = list(train_cfg.get("features", {}).get("cols", []))
    if not feature_cols:
        raise ValueError("features.cols must be specified")

    # Load dataset
    df = load_events_df(train_cfg, limit_files=args.limit_files)
    summary = summarize_events_df(df)

    # Labels
    df = df.copy()
    df["label"] = make_labels(df, train_cfg)

    # Splits
    train_df, val_df, test_df = make_splits(df, train_cfg)

    # Quality
    q_train = check_quality(train_df, label_col="label", feature_cols=feature_cols)
    q_val = check_quality(val_df, label_col="label", feature_cols=feature_cols)
    q_test = check_quality(test_df, label_col="label", feature_cols=feature_cols)

    # Features
    X_train = _select_features(train_df, feature_cols)
    y_train = train_df["label"].astype(str).to_numpy()

    X_val = _select_features(val_df, feature_cols)
    y_val = val_df["label"].astype(str).to_numpy()

    X_test = _select_features(test_df, feature_cols)
    y_test = test_df["label"].astype(str).to_numpy()

    model = _fit_logreg(train_cfg)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "features": ",".join(feature_cols),
                "val_days": train_cfg.get("split", {}).get("val_days", 0),
                "test_days": train_cfg.get("split", {}).get("test_days", 0),
            }
        )

        mlflow.log_dict(summary, "data_summary.json")
        mlflow.log_dict(asdict(q_train), "quality_train.json")
        mlflow.log_dict(asdict(q_val), "quality_val.json")
        mlflow.log_dict(asdict(q_test), "quality_test.json")

        t0 = time.time()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_time_s", time.time() - t0)

        all_labels = sorted(pd.Series(y_train).unique())

        val_metrics = None
        test_metrics = None

        if _has_rows(X_val):
            y_val_pred = model.predict(X_val)
            val_metrics = _eval_metrics(y_val, y_val_pred, all_labels)
            mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"])
            mlflow.log_dict(val_metrics, "val_metrics.json")

        if _has_rows(X_test):
            y_test_pred = model.predict(X_test)
            test_metrics = _eval_metrics(y_test, y_test_pred, all_labels)
            mlflow.log_metric("test_macro_f1", test_metrics["macro_f1"])

            if "DISTRACTED" in test_metrics["per_class"]:
                mlflow.log_metric(
                    "test_f1_distracted",
                    test_metrics["per_class"]["DISTRACTED"]["f1"],
                )

            mlflow.log_dict(test_metrics, "test_metrics.json")

        # Save model
        out_cfg = train_cfg.get("outputs", {})
        model_dir = Path(out_cfg.get("model_dir", "models"))
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "focusguard_logreg.joblib"
        joblib.dump({"model": model, "features": feature_cols}, model_path)
        mlflow.log_artifact(str(model_path))

        # Report
        report_dir = Path(out_cfg.get("report_dir", "reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "latest_eval.json"

        report = {
            "summary": summary,
            "quality": {
                "train": asdict(q_train),
                "val": asdict(q_val),
                "test": asdict(q_test),
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        print(f"DONE Wrote model → {model_path}")
        print(f"DONE Wrote report → {report_path}")


if __name__ == "__main__":
    main()
