"""compare_with_sklearn.py

Skrypt porównujący własne implementacje (DecisionTree, DecisionTreeREP, RandomForest)
z odpowiednikami z biblioteki scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import get_bc_dataset, get_wr_dataset, get_ww_dataset
from DecisionTree import DecisionTree, DecisionTreeREP
from RandomForest import RandomForest


def evaluate_models_on_dataset(get_dataset_func, models, n_experiments=30):
    _, _, y_tmp_train, y_tmp_test = get_dataset_func()
    labels = np.unique(np.concatenate([y_tmp_train, y_tmp_test]))

    results = {}

    for cls, model_name, params, is_sklearn in models:
        metrics_acc = []
        metrics_prec = []
        metrics_rec = []
        metrics_f1 = []
        cms = []

        for i in range(n_experiments):
            X_train, X_test, y_train, y_test = get_dataset_func(seed=i)

            if is_sklearn:
                model = cls(**params, random_state=i) if params else cls(random_state=i)
            else:
                model = cls(**params) if params else cls()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics_acc.append(accuracy_score(y_test, preds))
            metrics_prec.append(
                precision_score(y_test, preds, average="weighted", zero_division=0)
            )
            metrics_rec.append(
                recall_score(y_test, preds, average="weighted", zero_division=0)
            )
            metrics_f1.append(
                f1_score(y_test, preds, average="weighted", zero_division=0)
            )

            cm = confusion_matrix(y_test, preds, labels=labels)
            cms.append(cm)

        sum_cm = np.sum(cms, axis=0)
        avg_cm = sum_cm / float(n_experiments)

        results[model_name] = {
            "accuracy_mean": float(np.mean(metrics_acc)),
            "accuracy_std": float(np.std(metrics_acc)),
            "precision_mean": float(np.mean(metrics_prec)),
            "recall_mean": float(np.mean(metrics_rec)),
            "f1_mean": float(np.mean(metrics_f1)),
            "confusion_matrix_avg": np.round(avg_cm, 2),
            "confusion_matrix_sum": sum_cm.astype(int),
            "labels": labels,
        }

    return results


def print_dataset_comparison(dataset_name, eval_results):
    rows = []
    for model_name, res in eval_results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "precision_mean": res["precision_mean"],
                "recall_mean": res["recall_mean"],
                "f1_mean": res["f1_mean"],
            }
        )

    df = pd.DataFrame(rows).set_index("model")

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    print("\nSummary metrics (mean +/- std for accuracy):")
    df_display = df.copy()
    df_display["accuracy"] = df_display.apply(
        lambda r: f"{r['accuracy_mean']:.4f} (+/- {r['accuracy_std']:.4f})", axis=1
    )
    df_display = df_display[["accuracy", "precision_mean", "recall_mean", "f1_mean"]]
    df_display = df_display.rename(
        columns={
            "precision_mean": "precision",
            "recall_mean": "recall",
            "f1_mean": "f1",
        }
    )
    print(df_display.to_string())


def main():
    num_experiments = 30

    datasets = [
        (get_bc_dataset, "Breast Cancer"),
        (get_wr_dataset, "Wine Quality Red"),
        (get_ww_dataset, "Wine Quality White"),
    ]

    models = [
        # Własne implementacje
        (DecisionTree, "DecisionTree (own)", {}, False),
        (DecisionTreeREP, "DecisionTreeREP (own)", {}, False),
        (
            RandomForest,
            "RandomForest (own)",
            {"n_trees": 250, "max_features": "sqrt", "sample_size": 1.0},
            False,
        ),
        # sklearn implementacje
        (DecisionTreeClassifier, "DecisionTree (sklearn)", {"criterion": "entropy"}, True),
        (
            RandomForestClassifier,
            "RandomForest (sklearn)",
            {"n_estimators": 250, "max_features": "sqrt", "criterion": "entropy"},
            True,
        ),
    ]

    for get_func, name in datasets:
        results = evaluate_models_on_dataset(get_func, models, n_experiments=num_experiments)
        print_dataset_comparison(name, results)


if __name__ == "__main__":
    main()
