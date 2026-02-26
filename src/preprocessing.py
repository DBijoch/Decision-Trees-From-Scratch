import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

breast_cancer_data = pd.read_csv("data/wdbc.csv")
wine_quality_red_data = pd.read_csv("data/winequality-red.csv", sep=";")
wine_quality_white_data = pd.read_csv("data/winequality-white.csv", sep=";")


def discretize_dataset(
    data: pd.DataFrame, drop_columns: list[str]
) -> pd.DataFrame:
    X = data.drop(columns=drop_columns).copy()

    for col in X.columns:
        # Freedman-Diaconis
        q75, q25 = np.percentile(X[col], [75, 25])
        iqr = q75 - q25
        n = len(X[col])

        if iqr > 0:
            bin_width = 2 * iqr / (n ** (1 / 3))
            num_bins = int((X[col].max() - X[col].min()) / bin_width)
            num_bins = max(2, min(num_bins, 15))
        else:
            num_bins = 5

        discretizer = KBinsDiscretizer(
            n_bins=num_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        X[col] = discretizer.fit_transform(X[[col]]).flatten()

    return X


def split_train_test(
    data: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    test_size: float = 0.3,
    seed=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = discretize_dataset(data, drop_columns)
    y = data[target_column].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def get_bc_dataset(test_size: float = 0.3, seed=None):
    return split_train_test(
        breast_cancer_data,
        target_column="class",
        drop_columns=["id", "class"],
        test_size=test_size,
        seed=seed,
    )


def get_wr_dataset(test_size: float = 0.3, seed=None):
    return split_train_test(
        wine_quality_red_data,
        target_column="quality",
        drop_columns=["quality"],
        test_size=test_size,
        seed=seed,
    )


def get_ww_dataset(test_size: float = 0.3, seed=None):
    return split_train_test(
        wine_quality_white_data,
        target_column="quality",
        drop_columns=["quality"],
        test_size=test_size,
        seed=seed,
    )


if __name__ == "__main__":
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = split_train_test(
        breast_cancer_data,
        target_column="class",
        drop_columns=["id", "class"],
    )

    X_train_wr, X_test_wr, y_train_wr, y_test_wr = split_train_test(
        wine_quality_red_data,
        target_column="quality",
        drop_columns=["quality"],
    )

    X_train_ww, X_test_ww, y_train_ww, y_test_ww = split_train_test(
        wine_quality_white_data,
        target_column="quality",
        drop_columns=["quality"],
    )

    print("Breast Cancer Dataset:")
    print("X_train shape:", X_train_bc.shape)
    print("X_test shape:", X_test_bc.shape)
    print("y_train shape:", y_train_bc.shape)
    print("y_test shape:", y_test_bc.shape)
    

    print("\nWine Quality Red Dataset:")
    print("X_train shape:", X_train_wr.shape)
    print("X_test shape:", X_test_wr.shape)
    print("y_train shape:", y_train_wr.shape)
    print("y_test shape:", y_test_wr.shape)
    

    print("\nWine Quality White Dataset:")
    print("X_train shape:", X_train_ww.shape)
    print("X_test shape:", X_test_ww.shape)
    print("y_train shape:", y_train_ww.shape)
    print("y_test shape:", y_test_ww.shape)
    
