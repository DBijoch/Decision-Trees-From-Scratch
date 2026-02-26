import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime

from preprocessing import get_bc_dataset, get_wr_dataset, get_ww_dataset
from RandomForest import RandomForest


param_grid = {
    "n_trees": [300],
    "max_features": ["sqrt"],
    "sample_size": [0.6, 0.8, 1.0],
}

datasets = [
    (get_bc_dataset, "Breast Cancer"),
    (get_wr_dataset, "Wine Quality Red"),
    (get_ww_dataset, "Wine Quality White"),
]

num_cv_splits = 10


def convert_max_features(max_feat_param, n_features):
    """Konwertuje parametr max_features na int."""
    if max_feat_param == "sqrt":
        return int(np.sqrt(n_features))
    elif max_feat_param == "log2":
        return int(np.log2(n_features))
    elif isinstance(max_feat_param, float):
        return int(max_feat_param * n_features)
    else:
        return max_feat_param


def tune_dataset(get_dataset_func, dataset_name, n_cv_splits=5):
    """Testuje wszystkie kombinacje hiperparametrów dla danego datasetu."""
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters for: {dataset_name}")
    print(f"{'='*60}")

    X_tmp, _, _, _ = get_dataset_func()
    n_features = X_tmp.shape[1]

    results = []
    grid = list(ParameterGrid(param_grid))
    total_combinations = len(grid)

    for idx, params in enumerate(grid, 1):
        print(f"\nTesting combination {idx}/{total_combinations}: {params}")

        accuracies = []

        for split_idx in range(n_cv_splits):
            X_train, X_test, y_train, y_test = get_dataset_func(seed=split_idx)

            max_feat_for_model = params["max_features"]
            if isinstance(max_feat_for_model, float):
                max_feat_for_model = convert_max_features(
                    max_feat_for_model, n_features
                )

            rf = RandomForest(
                n_trees=params["n_trees"],
                max_features=max_feat_for_model,
                sample_size=params["sample_size"],
                random_state=split_idx,
            )

            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"  Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

        results.append(
            {
                "params": params,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "all_accuracies": accuracies,
            }
        )

    results.sort(key=lambda x: x["mean_accuracy"], reverse=True)

    return results


def print_top_results(results, dataset_name, top_n=5):
    """Wyświetla top N najlepszych wyników."""
    print(f"\n{'='*60}")
    print(f"Top {top_n} configurations for {dataset_name}:")
    print(f"{'='*60}")

    for idx, result in enumerate(results[:top_n], 1):
        print(
            f"\n#{idx} - Accuracy: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})"
        )
        print(f"Parameters:")
        for key, value in result["params"].items():
            print(f"  {key}: {value}")


def save_results_to_file(
    all_results, filename="hyperparameter_tuning_results.json"
):
    """Zapisuje wyniki do pliku JSON."""

    # Konwertuj numpy typy na Python natywne dla JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = {}
    for dataset_name, results in all_results.items():
        serializable_results[dataset_name] = []
        for result in results:
            serializable_results[dataset_name].append(
                {
                    "params": {
                        k: convert_numpy(v)
                        for k, v in result["params"].items()
                    },
                    "mean_accuracy": convert_numpy(result["mean_accuracy"]),
                    "std_accuracy": convert_numpy(result["std_accuracy"]),
                    "all_accuracies": [
                        convert_numpy(x) for x in result["all_accuracies"]
                    ],
                }
            )

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_cv_splits": num_cv_splits,
        "param_grid": {
            k: [convert_numpy(x) for x in v] for k, v in param_grid.items()
        },
        "results": serializable_results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to {filename}")


if __name__ == "__main__":
    all_results = {}

    for get_func, name in datasets:
        results = tune_dataset(get_func, name, n_cv_splits=num_cv_splits)
        all_results[name] = results
        print_top_results(results, name, top_n=5)

    save_results_to_file(all_results)

    print(f"\n\n{'='*60}")
    print("SUMMARY - Best configuration for each dataset:")
    print(f"{'='*60}")

    for dataset_name, results in all_results.items():
        best = results[0]
        print(f"\n{dataset_name}:")
        print(
            f"  Accuracy: {best['mean_accuracy']:.4f} (+/- {best['std_accuracy']:.4f})"
        )
        print(f"  Best params: {best['params']}")
