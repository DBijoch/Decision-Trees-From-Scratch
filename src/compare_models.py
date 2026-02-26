"""compare_models.py

Skrypt porównujący `DecisionTree`, "PrunedDecisionTree" (zaimplementowany jako
`DecisionTreeREP` w `DecisionTree.py`) oraz `RandomForest`.
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

from preprocessing import get_bc_dataset, get_wr_dataset, get_ww_dataset
from DecisionTree import DecisionTree, DecisionTreeREP
from RandomForest import RandomForest


def evaluate_models_on_dataset(get_dataset_func, models, n_experiments=30):
    _, _, y_tmp_train, y_tmp_test = get_dataset_func()
    labels = np.unique(np.concatenate([y_tmp_train, y_tmp_test]))

    results = {}

    for cls, model_name, params in models:
        metrics_acc = []
        metrics_prec = []
        metrics_rec = []
        metrics_f1 = []
        cms = []
        tree_nodes = []
        tree_leaves = []
        tree_depths = []

        for i in range(n_experiments):
            X_train, X_test, y_train, y_test = get_dataset_func(seed=i)

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
            
            # Collect tree statistics for DecisionTree models
            if hasattr(model, 'count_nodes'):
                tree_nodes.append(model.count_nodes())
                tree_leaves.append(model.count_leaves())
                tree_depths.append(model.get_depth())

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
        
        # Add tree statistics 
        if tree_nodes:
            results[model_name]["nodes_mean"] = float(np.mean(tree_nodes))
            results[model_name]["nodes_std"] = float(np.std(tree_nodes))
            results[model_name]["leaves_mean"] = float(np.mean(tree_leaves))
            results[model_name]["leaves_std"] = float(np.std(tree_leaves))
            results[model_name]["depth_mean"] = float(np.mean(tree_depths))
            results[model_name]["depth_std"] = float(np.std(tree_depths))

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
    # format accuracy with std inline
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

    # Tree statistics 
    print("\nTree Statistics:")
    tree_stats_rows = []
    for model_name, res in eval_results.items():
        if "nodes_mean" in res:
            tree_stats_rows.append({
                "model": model_name,
                "nodes": f"{res['nodes_mean']:.1f} (+/- {res['nodes_std']:.1f})",
                "leaves": f"{res['leaves_mean']:.1f} (+/- {res['leaves_std']:.1f})",
                "depth": f"{res['depth_mean']:.1f} (+/- {res['depth_std']:.1f})",
            })
    
    if tree_stats_rows:
        tree_df = pd.DataFrame(tree_stats_rows).set_index("model")
        print(tree_df.to_string())
    else:
        print("No tree statistics available (models without tree structure)")

    # Confusion matrices
    for model_name, res in eval_results.items():
        print("\n" + "-" * 60)
        print(f"Model: {model_name}")
        print("Labels:", res["labels"])
        print("Average confusion matrix (rounded):\n", res["confusion_matrix_avg"])
        print("Summed confusion matrix (integers):\n", res["confusion_matrix_sum"])


def main():
    num_experiments = 30

    datasets = [
        (get_bc_dataset, "Breast Cancer"),
        (get_wr_dataset, "Wine Quality Red"),
        (get_ww_dataset, "Wine Quality White"),
    ]

    models = [
        (DecisionTree, "DecisionTree", {}),
        (DecisionTreeREP, "PrunedDecisionTree", {}),
        (
            RandomForest,
            "RandomForest",
            {"n_trees": 250, "max_features": "sqrt", "sample_size": 1.0},
        ),
    ]

    for get_func, name in datasets:
        results = evaluate_models_on_dataset(get_func, models, n_experiments=num_experiments)
        print_dataset_comparison(name, results)


if __name__ == "__main__":
    main()
