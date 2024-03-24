import numpy as np
from data import load_mnist
import matplotlib.pyplot as plt


class Tensor2LayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def categorical_cross_entropy(actual, pred):
        return -np.sum(actual * np.log(pred)) / len(actual)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, X, y, learning_rate):
        m = len(X)

        output_error = self.a2 - y

        dW2 = (1 / m) * np.dot(self.a1.T, output_error)
        db2 = (1 / m) * np.sum(output_error, axis=0, keepdims=True)

        dh = np.dot(output_error, self.W2.T) * self.a1 * (1 - self.a1)

        dW1 = (1 / m) * np.dot(X.T, dh)
        db1 = (1 / m) * np.sum(dh, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(
        self, X, y, X_val=None, y_val=None, epochs=5, learning_rate=0.03, batch_size=64
    ):

        # collect losses
        train_loss, val_loss, batch_losses = [], [], []

        print("Start training ...")

        for epoch in range(epochs):
            print(f"epoch: {epoch}")

            # Shuffle the data
            permutation = np.random.permutation(y.shape[0])
            shuffled_X = X[permutation, :]
            shuffled_y = y[permutation, :]

            # Loop over the batches
            for i in range(0, y.shape[0], batch_size):
                batch_X = shuffled_X[i : i + batch_size]
                batch_y = shuffled_y[i : i + batch_size]

                # Forward pass
                output = self.forward(batch_X)

                # Backward pass
                self.backward(batch_X, batch_y, learning_rate)

                # Loss
                batch_loss = self.categorical_cross_entropy(batch_y, output)
                batch_losses.append(batch_loss)

            # Calculate training loss
            train_output = self.forward(X)
            train_loss.append(self.categorical_cross_entropy(y, train_output))

            if X_val is not None and y_val is not None:
                # Calculate validation loss
                val_output = self.forward(X_val)
                val_loss.append(self.categorical_cross_entropy(y_val, val_output))

        return train_loss, val_loss, batch_losses


# Load data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist(final=False)

# Normalize
xtrain = xtrain / 255.0
xval = xval / 255.0


# One-hot encode the labels
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i][label] = 1
    return encoded


y_train_one_hot = one_hot_encode(ytrain, num_cls)
y_val_one_hot = one_hot_encode(yval, num_cls)

# Hyperparameters
samples = xtrain.shape[0]

input_size = xtrain.shape[1]
hidden_size = 300
output_size = num_cls

learning_rate = 0.03
num_epochs = 5
batch_size = 64

# Create and train the neural network with mini-batch gradient descent
network = Tensor2LayerNetwork(input_size, hidden_size, output_size)


def print_model_performance(train_loss, val_loss):
    print("Model performance: ")
    print(f"Train loss | Val loss")
    for t_loss, v_loss in zip(train_loss, val_loss):
        print(f"{t_loss:1.04f} | {v_loss:1.04f}")


def plot_loss(losses, name, smoothing=True):
    if smoothing:
        smooth_loss = np.convolve(losses, np.ones(50), "valid") / 50

        plt.plot(losses, alpha=0.6, label="batch losses")
        plt.plot(smooth_loss, "--g", label="moving average")
    else:
        plt.plot(losses)

    plt.title(f"Training loss")
    plt.ylabel("loss")
    plt.xlabel("iterations")
    plt.legend()

    plt.savefig(f"{name}")
    plt.tight_layout()
    plt.show()


def plot_model_performance(training_losses, validation_losses, name):
    plt.plot(training_losses, "b-", label="training loss")
    plt.plot(validation_losses, "g--", label="validation loss")

    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f"{name}")
    plt.tight_layout()
    plt.show()


(train_loss, val_loss, batch_loss) = network.train(
    xtrain, y_train_one_hot, xval, y_val_one_hot, num_epochs, learning_rate, batch_size
)
