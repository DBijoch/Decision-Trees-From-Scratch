from DecisionTree import DecisionTree

import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=100, max_features="sqrt", sample_size=1.0, random_state=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees = []  # Lista krotek (drzewo, indeksy_cech)

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        # Convert input to numpy arrays
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.trees = []  

        if self.max_features == "sqrt":
            n_feat_select = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            n_feat_select = self.max_features
        else:
            n_feat_select = (
                n_features
            )

        n_feat_select = max(1, n_feat_select)

        for _ in range(self.n_trees):
            tree = DecisionTree()

            # Bootstrapping
            n_boot = int(n_samples * self.sample_size)
            idx_samples = np.random.choice(n_samples, n_boot, replace=True)

            # Feature sampling
            idx_features = np.random.choice(
                n_features, n_feat_select, replace=False
            )

            # Prepare data subset and train
            X_subset = X[idx_samples][:, idx_features]
            y_subset = y[idx_samples]

            # Train tree on subset
            tree.fit(X_subset, y_subset)

            # Store tree with selected feature indices for prediction
            self.trees.append((tree, idx_features))

    def predict(self, X):
        """Predict classes by majority voting from ensemble."""
        if hasattr(X, "values"):
            X = X.values
        X = np.array(X)

        # Collect predictions from each tree
        all_tree_predictions = []

        for tree, idx_features in self.trees:
            X_subset = X[:, idx_features]

            # Predict with single tree
            prediction = tree.predict(X_subset)
            all_tree_predictions.append(prediction)

        all_tree_predictions = np.array(all_tree_predictions)
        all_tree_predictions = all_tree_predictions.T

        final_predictions = []
        for sample_votes in all_tree_predictions:
            winner = Counter(sample_votes).most_common(1)[0][0]
            final_predictions.append(winner)

        return np.array(final_predictions)
