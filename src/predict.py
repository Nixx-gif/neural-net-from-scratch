from network import Network
from data import Data
import argparse


def main():
    """
    Load a trained model and evaluate it on a test set.

    Prints accuracy and binary cross-entropy loss to stdout.

    CLI arguments:
        --model      Path to the saved .pkl model file (required)
        --train_set  Path to the training CSV (required, used for scaler fitting)
        --test_set   Path to the test CSV (required)

    Example:
        python predict.py --model ../model/best_model.pkl \\
                          --train_set ../data/training_data.csv \\
                          --test_set ../data/test_data.csv
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved .pkl model file")
    parser.add_argument("--train_set", type=str, required=True,
                        help="Path to the training CSV (used to fit the scaler)")
    parser.add_argument("--test_set", type=str, required=True,
                        help="Path to the test CSV")
    args = parser.parse_args()

    net = Network.load(args.model)

    train_set = Data(args.train_set)
    test_set = Data(args.test_set)
    _, test_data = train_set.prepare_all_data2(train_set, test_set)

    accuracy = net.evaluate(test_data)
    bce_loss = net.binary_cross_entropy(test_data)

    print(f"Accuracy:              {accuracy}/{len(test_data)} ({100 * accuracy / len(test_data):.2f}%)")
    print(f"Binary Cross-Entropy:  {bce_loss:.6f}")


if __name__ == "__main__":
    main()