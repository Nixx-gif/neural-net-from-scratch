import numpy as np
import time
import random
import matplotlib.pyplot as plt
import pickle


class Network(object):
    """
    A fully-connected feedforward neural network trained with mini-batch SGD.

    The architecture uses sigmoid activations on all hidden layers and a softmax
    output layer. Backpropagation computes gradients analytically.

    Attributes:
        num_layers (int): Total number of layers including input and output.
        sizes (list[int]): Number of neurons per layer (e.g. [30, 24, 24, 2]).
        biases (list[np.ndarray]): Bias vectors for each non-input layer.
        weights (list[np.ndarray]): Weight matrices for each layer transition.
        history (dict): Training metrics recorded per epoch:
                        'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'.
    """

    def __init__(self, sizes):
        """
        Initialize the network with random weights and biases.

        Weights and biases are drawn from a standard normal distribution.

        Args:
            sizes (list[int]): Number of neurons in each layer, from input to output.
                               Example: [30, 24, 24, 2]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }

    def save(self, filename):
        """
        Serialize and save the full model to disk using pickle.

        Saves architecture (sizes), weights, biases, and training history.

        Args:
            filename (str): Path to the output .pkl file.
        """
        data = {
            'sizes': self.sizes,
            'weights': self.weights,
            'biases': self.biases,
            'history': self.history
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved: {filename}")

    @staticmethod
    def load(filename):
        """
        Load a previously saved model from disk.

        Reconstructs the Network instance and restores its weights,
        biases, and training history.

        Args:
            filename (str): Path to the .pkl file to load.

        Returns:
            Network: A fully restored Network instance.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        net = Network(data['sizes'])
        net.weights = data['weights']
        net.biases = data['biases']
        net.history = data['history']
        print(f"Model loaded: {filename}")
        print(f"Architecture: {data['sizes']}")
        return net

    def feedforward(self, a):
        """
        Compute the network output for a given input vector.

        Applies sigmoid activation on all hidden layers and softmax on the
        output layer.

        Args:
            a (np.ndarray): Input vector of shape (input_size, 1).

        Returns:
            np.ndarray: Output probability vector of shape (output_size, 1).
        """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def compute_loss(self, data):
        """
        Compute the mean squared error (MSE) loss over a dataset.

        Args:
            data (list[tuple]): List of (x, y) pairs where x is the input
                                vector and y is the one-hot target vector.

        Returns:
            float: Average MSE loss across all samples.
        """
        total_loss = 0.0
        for x, y in data:
            output = self.feedforward(x)
            total_loss += np.sum((output - y) ** 2) / 2
        return total_loss / len(data)

    def binary_cross_entropy(self, data):
        """
        Compute the binary cross-entropy (BCE) loss over a dataset.

        Formula: E = -1/N * sum(y * log(p) + (1 - y) * log(1 - p))

        A small epsilon is added to predictions to avoid log(0).

        Args:
            data (list[tuple]): List of (x, y) pairs where y is a one-hot
                                vector of shape (2, 1).

        Returns:
            float: Average BCE loss across all samples.
        """
        total_loss = 0.0
        epsilon = 1e-15
        for x, y in data:
            output = self.feedforward(x)
            output = np.clip(output, epsilon, 1 - epsilon)
            total_loss += -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output))
        return total_loss / len(data)

    def SGD(self, training_data, epochs, eta, mini_batch_size, test_data=None):
        """
        Train the network using mini-batch stochastic gradient descent.

        At each epoch the training data is shuffled and split into mini-batches.
        Gradients are computed via backpropagation and weights are updated.
        Loss and accuracy are recorded in self.history each epoch.

        Args:
            training_data (list[tuple]): List of (x, y) training pairs.
            epochs (int): Number of full passes over the training data.
            eta (float): Learning rate.
            mini_batch_size (int): Number of samples per mini-batch.
            test_data (list[tuple] | None): Optional evaluation set. If provided,
                                            test loss and accuracy are logged each epoch.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()

            train_loss = self.compute_loss(training_data)
            train_accuracy = 100.0 * self.evaluate(training_data) / n
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)

            if test_data:
                test_loss = self.compute_loss(test_data)
                test_accuracy = 100.0 * self.evaluate(test_data) / n_test
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_accuracy)
                print(f"Epoch {j+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}% | "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.2f}% "
                      f"({time2 - time1:.2f}s)")
            else:
                print(f"Epoch {j+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}% "
                      f"({time2 - time1:.2f}s)")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases using gradients from one mini-batch.

        Accumulates gradients over all samples in the batch via backpropagation,
        then applies the averaged gradient update scaled by the learning rate.

        Args:
            mini_batch (list[tuple]): List of (x, y) pairs for this batch.
            eta (float): Learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, X, y):
        """
        Compute gradients of the loss with respect to all weights and biases.

        Performs a forward pass to collect activations and pre-activations (z),
        then propagates the error backward through the network.

        Output layer uses softmax. Hidden layers use sigmoid.

        Args:
            X (np.ndarray): Input vector of shape (input_size, 1).
            y (np.ndarray): One-hot target vector of shape (output_size, 1).

        Returns:
            tuple:
                - nabla_b (list[np.ndarray]): Gradients for each bias vector.
                - nabla_w (list[np.ndarray]): Gradients for each weight matrix.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = X
        activations = [activation]
        zs = []

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def cost_derivative(self, output, y):
        """
        Compute the derivative of the MSE cost with respect to the output.

        Args:
            output (np.ndarray): Network output vector.
            y (np.ndarray): Target one-hot vector.

        Returns:
            np.ndarray: Element-wise difference (output - y).
        """
        return (output - y)

    def evaluate(self, test_data):
        """
        Count the number of correctly classified samples in a dataset.

        Handles two label formats:
        - One-hot vector (np.ndarray): argmax extracts the class index.
        - Scalar integer: used directly as the target class.

        Args:
            test_data (list[tuple]): List of (x, y) pairs.

        Returns:
            int: Number of correctly predicted samples.
        """
        test_results = []
        for x, y in test_data:
            prediction = np.argmax(self.feedforward(x))
            target = np.argmax(y) if isinstance(y, np.ndarray) else int(y)
            test_results.append((prediction, target))
        return sum(int(pred == target) for pred, target in test_results)

    def plot_history(self):
        """
        Plot and save training loss and accuracy curves.

        Generates a side-by-side figure with:
        - Left panel: train and test loss per epoch (MSE)
        - Right panel: train and test accuracy per epoch (%)

        The figure is saved as 'training_curves.png' in the current directory.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['test_loss']:
            ax1.plot(epochs, self.history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_title('Loss over training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        if self.history['test_accuracy']:
            ax2.plot(epochs, self.history['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_title('Accuracy over training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Plot saved: training_curves.png")
        plt.show()


def sigmoid(z):
    """
    Compute the sigmoid activation function element-wise.

    Args:
        z (np.ndarray): Pre-activation values.

    Returns:
        np.ndarray: Values mapped to range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function element-wise.

    Args:
        z (np.ndarray): Pre-activation values.

    Returns:
        np.ndarray: sigmoid(z) * (1 - sigmoid(z))
    """
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """
    Compute the softmax function over a vector.

    Subtracts the max value before exponentiation for numerical stability.

    Args:
        z (np.ndarray): Raw output vector of shape (n, 1).

    Returns:
        np.ndarray: Probability distribution summing to 1.
    """
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z))