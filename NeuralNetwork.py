import numpy as np
from matplotlib import pyplot as plt


class NeuralNet:
    """
    3 layered NN
    """

    def __init__(self, layers=None, learning_rate=0.001, iterations=100):
        if layers is None:
            layers = [4, 2, 1]
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def set_init_weights(self):
        """
        sets the initial weights of each layer
        :return: updates the self.parameter dictionary with all weights nd biases
        """
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )

    def relu(self, Z):
        """
        rectified linear activation function to return 0 if net input is below 0 and otherwise linear activation
        :param Z: net input z of node
        :return: the rectified linear activation
        """
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        """
        sigmoid output function for binary classification
        :param Z: net input of node
        :return: approximated output value of the output layer for each node
        """
        return 1 / (1 + np.exp(-Z))

    def eta(self, x):
        """
        returning small value ETA to prevent the log of 0
        :param x: y_hat
        :return: either y_hat or the ETA value (whichever is bigger)
        """
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def entropy_loss(self, y, yhat):
        """
        cross entropy loss function for updating weights
        :param y: actual value y
        :param yhat: predicted value y hat
        :return: the loss of the current model compared to actual gold label
        """
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat)
        yhat_inv = self.eta(yhat_inv)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
        """
        forward pass through the network
        :return: the predicted value and the current loss of the model
        """
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']

        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss

    def dRelu(self, x):
        """

        :param x:
        :return:
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def backpropagation(self, yhat):
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = yhat * yhat_inv
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.set_init_weights()

        for i in range(self.iterations):
            y_hat, loss = self.forward_propagation()
            self.backpropagation(y_hat)
            self.loss.append(loss)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        prediction = self.sigmoid(Z2)
        return np.round(prediction)

    def acc(self, y, yhat):
        accuracy = int(np.sum(y == yhat) / len(y) * 100)
        return accuracy

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()