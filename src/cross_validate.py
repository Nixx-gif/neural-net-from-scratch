from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from network import Network
from data import Data
import numpy as np


def cross_validate(filepath, k_folds=5):
    """
    Evaluate the network using stratified k-fold cross-validation.

    For each fold, the scaler is fit on the training split only to prevent
    data leakage. A fresh network is trained from scratch on each fold.
    Results (accuracy per fold, mean, std, min, max) are printed to stdout.

    Args:
        filepath (str): Path to the raw CSV dataset.
        k_folds (int): Number of folds (default: 5).

    Returns:
        list[float]: Accuracy (%) for each fold.
    """
    prep = Data(filepath)

    diagnosis_map = {'M': 1, 'B': 0}
    y = prep.df.iloc[:, 1].map(diagnosis_map).values
    X = prep.df.iloc[:, 2:32].values

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{k_folds}")
        print(f"{'='*60}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        training_data = [
            (x.reshape(-1, 1),
             np.array([[1.0], [0.0]]) if label == 0 else np.array([[0.0], [1.0]]))
            for x, label in zip(X_train, y_train)
        ]
        test_data = [
            (x.reshape(-1, 1), label)
            for x, label in zip(X_test, y_test)
        ]

        net = Network([30, 16, 2])
        net.SGD(training_data, epochs=100, eta=0.5, mini_batch_size=16, test_data=None)

        accuracy = net.evaluate(test_data) / len(test_data) * 100
        accuracies.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.2f}%")

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({k_folds} folds)")
    print(f"{'='*60}")
    print(f"Mean:   {np.mean(accuracies):.2f}%")
    print(f"Std:    {np.std(accuracies):.2f}%")
    print(f"Min:    {np.min(accuracies):.2f}%")
    print(f"Max:    {np.max(accuracies):.2f}%")

    return accuracies


if __name__ == "__main__":
    cross_validate('../data/data.csv', k_folds=10)