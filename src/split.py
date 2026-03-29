import pandas as pd
import argparse
from data import Data
from sklearn.model_selection import train_test_split


def main():
    """
    Split a raw CSV dataset into training and test CSV files.

    The split is stratified on the Diagnosis column to preserve class balance.
    Output files are written to the current directory as training_data.csv
    and test_data.csv.

    CLI arguments:
        --csv   Path to the input CSV file (required)
        --ratio Test set proportion, between 0 and 1 (default: 0.3)
        --out   Output directory for the split files (default: ../data)

    Example:
        python split.py --csv ../data/data.csv --ratio 0.2 --out ../data
    """
    parser = argparse.ArgumentParser(description="Split a dataset into train/test CSV files")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Test set proportion (default: 0.3)")
    parser.add_argument("--out", type=str, default="../data",
                        help="Output directory for the split files (default: ../data)")
    args = parser.parse_args()

    data = Data(args.csv)
    X = data.df.drop(columns=['Diagnosis_encoded']).values

    training_data, test_data = train_test_split(
        X,
        test_size=args.ratio,
        random_state=42,
        stratify=X[:, 1]
    )

    df_train = pd.DataFrame(training_data)
    df_test = pd.DataFrame(test_data)

    train_path = f"{args.out}/training_data.csv"
    test_path = f"{args.out}/test_data.csv"

    df_train.to_csv(train_path, index=False, header=False)
    df_test.to_csv(test_path, index=False, header=False)

    print(f"Train set: {len(df_train)} samples -> {train_path}")
    print(f"Test set:  {len(df_test)} samples -> {test_path}")


if __name__ == "__main__":
    main()