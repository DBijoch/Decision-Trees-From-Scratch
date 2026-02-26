import numpy as np
import warnings
from preprocessing import (
    get_bc_dataset,
    get_wr_dataset,
    get_ww_dataset,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message="Bins whose width are too small")


class DecisionTree:
    def __init__(self):
        self.tree = None

    def _entropy(self, y):
        """Calculate the entropy of label array y."""
        value, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy

    def _information_gain(self, X_column, y):
        """Calculate information gain for a feature split."""
        total_entropy = self._entropy(y)
        n_samples = len(y)

        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = np.sum(
            [
                (count / n_samples) * self._entropy(y[X_column == value])
                for value, count in zip(values, counts)
            ]
        )

        return total_entropy - weighted_entropy

    def _best_split(self, X, y):
        """Find the best feature to split on."""
        n_features = X.shape[1]
        best_gain = -1
        best_feature = None

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            gain = self._information_gain(X_column, y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index

        return best_feature

    def _build_tree(self, X, y):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # If all samples belong to the same class, return that class
        if len(unique_classes) == 1:
            return unique_classes[0]

        # If there are no features left to split on, return the majority class
        if n_features == 0:
            return unique_classes[np.argmax(class_counts)]

        # Find the best feature to split on
        best_feature = self._best_split(X, y)
        if best_feature is None:
            return unique_classes[np.argmax(class_counts)]

        tree = {best_feature: {}}
        feature_values = np.unique(X[:, best_feature])

        for value in feature_values:
            indices = X[:, best_feature] == value
            X_subset = X[indices][:, np.arange(n_features) != best_feature]
            y_subset = y[indices]

            subtree = self._build_tree(X_subset, y_subset)
            tree[best_feature][value] = subtree
        tree[best_feature]["__default__"] = unique_classes[
            np.argmax(class_counts)
        ]

        return tree

    def fit(self, X, y):
        """Fit the decision tree to the training data."""
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, tree):
        """Predict the class label for a single sample."""
        if not isinstance(tree, dict):
            return tree

        feature_index = next(iter(tree))
        feature_value = sample[feature_index]

        if feature_value in tree[feature_index]:
            subtree = tree[feature_index][feature_value]
            reduced_sample = np.delete(sample, feature_index)
            return self._predict_sample(reduced_sample, subtree)
        return tree[feature_index]["__default__"]

    def predict(self, X):
        """Predict class labels for the input samples."""
        if hasattr(X, "values"):
            X = X.values
        X = np.array(X)
        return np.array(
            [self._predict_sample(sample, self.tree) for sample in X]
        )

    def count_nodes(self, tree=None):
        """Count total number of nodes in the tree (internal nodes + leaves)."""
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return 1

        # Internal node
        feature_index = next(iter(tree))
        count = 1  # Count this internal node

        for key, subtree in tree[feature_index].items():
            if key != "__default__":
                count += self.count_nodes(subtree)

        return count

    def count_leaves(self, tree=None):
        """Count number of leaf nodes in the tree."""
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return 1

        # Internal node
        feature_index = next(iter(tree))
        count = 0

        for key, subtree in tree[feature_index].items():
            if key != "__default__":
                count += self.count_leaves(subtree)

        return count

    def get_depth(self, tree=None):
        """Get maximum depth of the tree."""
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return 0

        # Internal node
        feature_index = next(iter(tree))
        max_depth = 0

        for key, subtree in tree[feature_index].items():
            if key != "__default__":
                depth = self.get_depth(subtree)
                max_depth = max(max_depth, depth)

        return max_depth + 1


class DecisionTreeREP(DecisionTree):
    """Decision Tree with Reduced Error Pruning (REP)."""

    def __init__(self):
        super().__init__()

    def _get_subtree_errors(self, tree, X, y):
        """Calculate number of errors for a subtree on validation data."""
        if len(X) == 0:
            return 0
        preds = np.array([self._predict_sample(sample, tree) for sample in X])
        return np.sum(preds != y)

    def _prune(self, tree, X, y):
        """Recursively prune tree using Reduced Error Pruning (REP)."""
        if not isinstance(tree, dict):
            return tree

        feature_index = next(iter(tree))

        if len(y) == 0:
            return tree

        for value in list(tree[feature_index].keys()):
            if value == "__default__":
                continue

            indices = X[:, feature_index] == value
            X_subset = X[indices]
            y_subset = y[indices]

            mask_cols = np.arange(X.shape[1]) != feature_index
            X_subset_reduced = X_subset[:, mask_cols]

            tree[feature_index][value] = self._prune(
                tree[feature_index][value], X_subset_reduced, y_subset
            )

        tree_errors = self._get_subtree_errors(tree, X, y)

        default_class = tree[feature_index]["__default__"]
        leaf_errors = np.sum(y != default_class)

        if leaf_errors <= tree_errors:
            return default_class

        return tree

    def fit(self, X, y, val_test_size=0.2):
        """Fit and prune the decision tree using REP strategy."""
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        X = np.array(X)
        y = np.array(y)

        x_tr, x_val, ytr, y_val = train_test_split(X, y, test_size=val_test_size)

        self.tree = self._build_tree(x_tr, ytr)

        self.tree = self._prune(self.tree, x_val, y_val)
