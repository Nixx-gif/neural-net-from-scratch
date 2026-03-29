import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


class Data(object):
    """
    Data loader and preprocessor for the Wisconsin Breast Cancer dataset.

    Handles CSV loading, feature/label extraction, train/test splitting,
    normalization, and vectorization into network-ready format.
    Also provides EDA utilities: boxplots, correlation heatmap, class distribution.

    Attributes:
        df (pd.DataFrame): Raw loaded dataframe with named columns.
    """

    def __init__(self, data_path):
        """
        Load a CSV file and encode the diagnosis column.

        Expects a headerless CSV with columns:
        [ID, Diagnosis, Feature_1, ..., Feature_30]

        Diagnosis values 'M' (malignant) and 'B' (benign) are encoded as 1 and 0.

        Args:
            data_path (str): Path to the CSV file.
        """
        column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
        self.df = pd.read_csv(data_path, names=column_names, header=None)
        self.df['Diagnosis_encoded'] = self.df.iloc[:, 1].map({'M': 1, 'B': 0})

    def load_data_wrapper(self):
        """
        Extract raw features and encoded labels as numpy arrays.

        Returns:
            tuple:
                - X (np.ndarray): Feature matrix of shape (n_samples, 30).
                - y (np.ndarray): Label vector of shape (n_samples,) with values 0 or 1.
        """
        X = self.df.iloc[:, 2:32].values
        y = self.df['Diagnosis_encoded'].values
        return X, y

    def prepare_all_data(self, features, labels):
        """
        Split features and labels into train/test sets, normalize, and vectorize.

        Uses a 70/30 train/test split with stratification.
        Scaler is fit on the training set only to prevent data leakage.

        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, 30).
            labels (np.ndarray): Label vector of shape (n_samples,).

        Returns:
            tuple:
                - training_data (list[tuple]): Vectorized training samples.
                - test_data (list[tuple]): Vectorized test samples.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        training_data = vectorize_data(X_train, y_train)
        test_data = vectorize_data(X_test, y_test)

        return training_data, test_data

    def prepare_all_data2(self, train_set, test_set):
        """
        Normalize and vectorize pre-split train and test Data objects.

        Loads raw arrays from both Data instances, fits the scaler on the
        training set only, and returns both sets in vectorized format.

        Args:
            train_set (Data): Data instance for the training split.
            test_set (Data): Data instance for the test split.

        Returns:
            tuple:
                - training_data (list[tuple]): Vectorized training samples.
                - test_data (list[tuple]): Vectorized test samples.
        """
        X_train, y_train = train_set.load_data_wrapper()
        X_test, y_test = test_set.load_data_wrapper()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        training_data = vectorize_data(X_train, y_train)
        test_data = vectorize_data(X_test, y_test)

        return training_data, test_data

    def info(self):
        """Print a concise summary of the dataframe (dtypes, non-null counts)."""
        print(self.df.info())

    def head(self):
        """Print the first 5 rows of the dataframe."""
        print(self.df.head())

    def miss_val(self):
        """Print the count of missing values per column."""
        print(self.df.isnull().sum())

    def describe(self):
        """Print descriptive statistics for all numerical columns."""
        print(self.df.describe())

    def boxplots(self):
        """
        Display boxplots for all features, grouped in sets of 10 per figure.

        Useful for spotting outliers and comparing feature distributions.
        """
        features = self.df.iloc[:, 2:]
        n_features_per_plot = 10
        n_plots = (len(features.columns) + n_features_per_plot - 1) // n_features_per_plot

        for plot_num in range(n_plots):
            start_idx = plot_num * n_features_per_plot
            end_idx = min(start_idx + n_features_per_plot, len(features.columns))
            cols_subset = features.columns[start_idx:end_idx]

            fig, axs = plt.subplots(len(cols_subset), 1, figsize=(10, 20), dpi=95)
            if len(cols_subset) == 1:
                axs = [axs]
            for i, col in enumerate(cols_subset):
                axs[i].boxplot(features[col].dropna(), vert=False)
                axs[i].set_ylabel(f'Feature {start_idx + i + 1}', fontsize=9)
                axs[i].set_title(f'Feature {start_idx + i + 1}', fontsize=10)

            plt.tight_layout()
            plt.suptitle(f'Boxplots - Group {plot_num + 1}', y=1.001)
            plt.show()

    def corr_analysis(self):
        """
        Display a heatmap of the top 16 features most correlated with diagnosis.

        Computes the absolute Pearson correlation between each feature and the
        encoded diagnosis label, then plots the top 16 as a symmetric heatmap.
        Also prints the ranked list with color-coded correlation strength.
        """
        data_tmp = self.df.copy()
        features = data_tmp.iloc[:, 2:].copy()
        features['Diagnosis'] = data_tmp['Diagnosis_encoded']

        corr = features.corr()
        corr_with_diagnosis = corr['Diagnosis'].abs().sort_values(ascending=False)
        top_features = corr_with_diagnosis.head(16).index.tolist()
        corr_subset = corr.loc[top_features, top_features]

        plt.figure(figsize=(18, 16), dpi=100)
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, linewidths=2,
                    cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1,
                    annot_kws={"fontsize": 10})
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.title('Top 16 Features — Correlation with Diagnosis',
                  fontsize=16, pad=20, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("\nTop 16 features correlated with Diagnosis:")
        print("=" * 60)
        for i, (feature, value) in enumerate(corr_with_diagnosis.head(16).items(), 1):
            if feature != 'Diagnosis':
                marker = "HIGH" if value > 0.7 else "MED" if value > 0.5 else "LOW"
                print(f"{i:2}. [{marker}] {feature}: {value:.4f}")

    def pie(self):
        """
        Display a pie chart showing the class distribution (Malignant vs Benign).
        """
        self.df['Diagnosis'].value_counts().plot.pie(
            labels=['M', 'B'], autopct='%.f%%', shadow=True
        )
        plt.title('Diagnosis Distribution')
        plt.show()


def vectorize_data(X, y):
    """
    Convert feature matrix and label array into a list of (x_vec, y_vec) tuples.

    Each sample is reshaped into a column vector. Labels are one-hot encoded
    into vectors of shape (2, 1): index 0 = Benign, index 1 = Malignant.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, 30).
        y (np.ndarray): Integer label array with values 0 (benign) or 1 (malignant).

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Network-ready list of (x_vec, y_vec) pairs.
    """
    data = []
    for feature, label in zip(X, y):
        x_vec = feature.reshape(30, 1)
        y_vec = np.zeros((2, 1))
        y_vec[label] = 1.0
        data.append((x_vec, y_vec))
    return data