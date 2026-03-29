from network import Network
from data import Data
import argparse


def main():
    """
    Entry point for training the multilayer perceptron.

    Parses CLI arguments, loads and preprocesses train/test datasets,
    trains the network using SGD, saves the best model, then plots
    the training history.

    CLI arguments:
        --layer         Hidden layer sizes (default: [24, 24, 24])
        --epochs        Number of training epochs (default: 100)
        --loss          Loss function name (default: categoricalCrossentropy)
        --batch_size    Mini-batch size (default: 16)
        --learning_rate Learning rate (default: 0.1)
        --train_set     Path to the training CSV file (required)
        --test_set      Path to the test CSV file (required)

    Example:
        python main.py --train_set ../data/training_data.csv \\
                        --test_set ../data/test_data.csv \\
                        --layer 24 24 --epochs 100 --learning_rate 0.1
    """
    parser = argparse.ArgumentParser(description="Multilayer Perceptron — breast cancer classifier")
    parser.add_argument("--layer", type=int, nargs="*", default=[24, 24, 24],
                        help="Hidden layer sizes. Input (30) and output (2) are fixed.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy",
                        help="Loss function (default: categoricalCrossentropy)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch size (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--train_set", type=str, required=True,
                        help="Path to the training CSV dataset")
    parser.add_argument("--test_set", type=str, required=True,
                        help="Path to the test CSV dataset")

    args = parser.parse_args()

    train_set = Data(args.train_set)
    test_set = Data(args.test_set)

    training_data, test_data = train_set.prepare_all_data2(train_set, test_set)

    net = Network([30] + args.layer + [2])

    net.SGD(training_data, args.epochs, args.learning_rate, args.batch_size, test_data=test_data)

    net.save('../model/best_model.pkl')

    loaded_net = Network.load('../model/best_model.pkl')
    loaded_net.plot_history()


if __name__ == "__main__":
    main()