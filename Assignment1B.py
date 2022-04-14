import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class preprocessing:

    def __init__(self, filepath):
        self.df = self.reader(filepath)

    @staticmethod
    def reader(filepath):
        """
        reads a csv file
        :param filepath: path to the file
        :return: only reads the columns we want
        """
        input_vars = ['income', 'social_fear', 'improve_yourself_how', 'friends', 'depressed']
        return pd.DataFrame(pd.read_csv(filepath, delimiter=",", usecols=input_vars))

    def cleaning_improvement(self):
        """
        Clenas the improve_yourself_how column. it counts the commas in an answer
        :return: the number of self-improvements of a person (none, 1, 2+)
        """
        for answer in self.df['improve_yourself_how']:
            occurrence = answer.count(',')

            if occurrence == 1:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=int(1))
            elif occurrence >= 2:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=int(2))
            else:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=int(0))

            self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                      value=int(occurrence))
        self.df.rename(columns={'improve_yourself_how': 'number_of_self_improvements'}, inplace=True)

    def cleaning_yes_no(self, column_name):
        """
        cleans the column by converting categrocial data into binary
        :param column_name: the column that should be processed
        :return: binary values in the column and updates the df
        """
        df_one = pd.get_dummies(self.df[column_name])
        df_one = df_one.drop(['No'], axis=1)
        df_one = df_one.rename(columns={"Yes": column_name})

        self.df = self.df.drop(column_name, axis=1)
        self.df = pd.concat((df_one, self.df), axis=1)

    def cleaning_income(self):
        """
        takes the income as a string
        :return: return the max. income as an integer
        """
        for i in self.df['income']:
            digits = ''.join(filter(lambda i: i.isdigit(), i[-7:]))
            if len(digits) != 0:
                self.df['income'] = self.df['income'].replace(to_replace=i, value=float(digits))
            else:
                self.df['income'] = self.df['income'].replace(to_replace=i, value=float(200000))


class NeuralNet:

    def __init__(self, layers=[4, 2, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def set_init_weights(self):
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def eta(self, x):
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat)
        yhat_inv = self.eta(yhat_inv)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
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
        self.set_init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.backpropagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        accuracy = int(np.sum(y == yhat) / len(y) * 100)
        return accuracy

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()


path = 'data/forever_alone.csv'
preprocessed_file = preprocessing(path)
preprocessed_file.cleaning_improvement()
preprocessed_file.cleaning_yes_no('depressed')
preprocessed_file.cleaning_yes_no('social_fear')
preprocessed_file.cleaning_income()

# CHECKING IF THERES NO MISSING VALUES ALL TYPES ARE THE SAME, ETC.

# print(preprocessed_file.df.head())
# print(preprocessed_file.df.shape)
# print(preprocessed_file.df.isna().sum())
# print(preprocessed_file.df.dtypes)


# SPLITTING DATAFRAME INTO TRAINING AND TESTING SETS

X = preprocessed_file.df.drop(columns=['depressed'])
y_label = preprocessed_file.df['depressed'].values.reshape(X.shape[0], 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

# STANDARDIZING THE INPUT SETS

sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

# CHECKING SHAPES OF THE SETS

# print(f"Shape of train set is {Xtrain.shape}")
# print(f"Shape of test set is {Xtest.shape}")
# print(f"Shape of train label is {ytrain.shape}")
# print(f"Shape of test labels is {ytest.shape}")

# TRAINING THE NEURAL NET

nn = NeuralNet()
nn.fit(Xtrain, ytrain)
# nn.plot_loss()

train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

training_accuracy = nn.acc(ytrain, train_pred)
testing_accuracy = nn.acc(ytest, test_pred)

print(f'The training accuracy is: {training_accuracy}')
print(f'The testing accuracy is: {testing_accuracy}')
