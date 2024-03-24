import math
import random
import matplotlib.pyplot as plt

from data import load_synth, load_mnist

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def softmax(z):
    exp = [math.exp(sm) for sm in z]
    return [ex / sum(exp) for ex in exp]


class PurePython2layerNetwork:
    def __init__(self):
        self.input_nodes = 2
        self.hidden_nodes = 3
        self.target_nodes = 2

        self.W1 = [[1., 1., 1.], [-1., -1., -1.]]
        self.b1 = [0., 0., 0.]
        self.W2 = [[1., 1.], [-1., -1.], [-1., -1.]]
        self.b2 = [0., 0.]

    def init_weights(self):
        self.W1 = [[random.normalvariate() for _ in range(self.hidden_nodes)] for _ in range(self.input_nodes)]
        self.b1 = [0 for _ in range(self.hidden_nodes)]
        self.W2 = [[random.normalvariate() for _ in range(self.target_nodes)] for _ in range(self.hidden_nodes)]
        self.b2 = [0 for _ in range(self.target_nodes)]

    def sigmoid(self, z):
        1 / (1 + math.exp(-z))

    def softmax(self, z):
        exp = [math.exp(sm) for sm in z]
        return [ex / sum(exp) for ex in exp]

    def cross_entropy(self, y, y_hat):
        loss = 0
        for i in range(len(y)):
            loss -= y[i] * math.log(y_hat[i])
        return loss

    def forward_propagation(self, inputs):
        input_to_hidden = [sum(self.W1[i][j] * inputs[i] for i in range(self.input_nodes)) + self.b1[j] for j in
                           range(self.hidden_nodes)]
        hidden_output = [sigmoid(z) for z in input_to_hidden]
        hidden_to_output = [sum(self.W2[i][j] * hidden_output[i] for i in range(self.hidden_nodes)) + self.b2[j] for j
                            in range(self.target_nodes)]
        pred_output = softmax(hidden_to_output)

        return hidden_output, pred_output

    def backpropagation(self, x, y, pred, hidden_output):

        loss_error = [pred[t] - y[t] for t in range(self.target_nodes)]

        dW2 = [[loss_error[i] * hidden_output[j] for i in range(self.target_nodes)] for j in range(self.hidden_nodes)]
        db2 = loss_error

        dh = []
        for i, h in enumerate(hidden_output):
            a = h * (1 - h) * sum([loss_error[j] * self.W2[i][j] for j in range(self.target_nodes)])
            dh.append(a)

        dW1 = [[dh[i] * x[j] for i in range(self.hidden_nodes)] for j in range(self.input_nodes)]
        db1 = dh

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, lr=0.02):

        # input-to-hidden
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                self.W1[i][j] -= lr * dW1[i][j]

        for i in range(self.hidden_nodes):
            self.b1[i] -= lr * db1[i]

        # hidden-to-output
        for i in range(self.hidden_nodes):
            for j in range(self.target_nodes):
                self.W2[i][j] -= lr * dW2[i][j]

        for i in range(self.target_nodes):
            self.b2[i] -= lr * db2[i]

    def train_network(self, data, epochs=10, lr=0.1):
        print('Start training ...')
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(data)

            for instance in range(len(data)):
                hidden_output, pred = self.forward_propagation(data[instance][0])
                total_loss += self.cross_entropy(data[instance][1], pred)
                dW1, db1, dW2, db2 = self.backpropagation(x=data[instance][0], y=data[instance][1], pred=pred,
                                                          hidden_output=hidden_output)
                self.update_weights(dW1, db1, dW2, db2, lr)

            losses.append(total_loss)

            print(f"Epoch {epoch + 1}, Loss: {total_loss}")

        self.plot_losses(losses)

    def plot_losses(self, loss):
        plt.plot(loss)
        plt.title('Training loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.show()


X = [1., -1.]
y = [1., 0.]

q3_network = PurePython2layerNetwork()

hidden_output, pred_output = q3_network.forward_propagation(inputs=X)

dW1, db1, dW2, db2 = q3_network.backpropagation(x=X, y=y, pred=pred_output, hidden_output=hidden_output)

dW1 = [[round(w, 5) for w in l] for l in dW1]
dW2 = [[round(w, 5) for w in l] for l in dW2]

print(f"Question 3 | Do a single forward pass and backpropagation\n")
print(f"Derivatives w.r.t W1 & b1: {dW1, db1}")
print(f"Derivatives w.r.t W2 & b2: {dW2, db2}")


## Question 4

(xtrain, ytrain), (xval, yval), num_cls = load_synth(num_train=20000)

q4_network = PurePython2layerNetwork()

def normalize_data(X):
    norm_data = []

    feature_1 = []
    feature_2 = []

    for i in X:
        feature_1.append(i[0])
        feature_2.append(i[1])

    min_feature_1 = min(feature_1)
    max_feature_1 = max(feature_1)
    min_feature_2 = min(feature_2)
    max_feature_2 = max(feature_2)

    for i in X:
        scaled_1 = (i[0] - min_feature_1) / (max_feature_1 - min_feature_1)
        scaled_2 = (i[1] - min_feature_2) / (max_feature_2 - min_feature_2)

        norm_data.append([scaled_1, scaled_2])

    return norm_data


ytrain = [[1 - i, i] for i in ytrain]
xtrain = normalize_data(xtrain)

data_pairs = [(xtrain[i], ytrain[i]) for i in range(len(xtrain))]

q4_network.init_weights()

print(f"\nQuestion 4 | Show a decreasing training loss \n")
q4_network.train_network(data_pairs, epochs=20)