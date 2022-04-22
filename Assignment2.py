import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNet
from RegressionClassifier import Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv


class TitanicPreProcessing:

    def __init__(self, filename, plotting):
        self.df = pd.DataFrame(pd.read_csv(filename, delimiter=","))
        # self.test_data = pd.DataFrame(pd.read_csv('data/titanic/test.csv', delimiter=","))

        # print(self.train_data.head())
        # print(self.train_data.info())
        # print(self.train_data.isna().sum())

        # BECAUSE THE COLUMN "CABIN" HAS 687 MISSING VALUES WE WILL OMIT IT FROM THE ANALYSIS
        # BECAUSE NAME DOES NOT HAVE A SIGNIFICIANT CORRELATION, IT WILL BE OMITTED TOO
        # TICKET DOES NOT SEEM TO BE A CONSISTENT PREDICTOR FOR SURVIVAL THEREFORE ALSO DELETED

        del self.df['Cabin']
        del self.df['Name']
        del self.df['Ticket']

        self.nan_replacement()
        self.numerical()

        if plotting:
            self.eda_categorical_plotting()

    def eda_categorical_plotting(self):
        columns = ["Pclass", "Sex", "SibSp", "Parch", 'Embarked', 'Age', 'Fare']

        for col in columns:
            fig = px.histogram(self.df, x=col, y='Survived', title=col)
            fig.update_layout(bargap=0.2)
            fig.write_image(f"figures/titanic/{col}.png")
            # fig.show()

    def numerical(self):
        columns = ["Sex", 'Embarked']

        for col in columns:
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(self.df[col])
            self.df[col] = label_encoder.transform(self.df[col])

    def nan_replacement(self):
        columns = {'Age': str(30), 'Embarked': str('S'), 'Fare': str(14.5)}
        for col in columns.keys():
            self.df[col].fillna(columns[col], inplace=True)

    def df_return(self):
        if 'Survived' in self.df.columns:
            X = self.df.drop(columns=['Survived'])
            y = self.df['Survived'].values.reshape(X.shape[0], 1)
            return X, y
        else:
            return self.df


def create_output_csv(filepath, survived: list):
    df_id = pd.DataFrame(pd.read_csv('data/titanic/test.csv', delimiter=',', usecols=['PassengerId']))
    survived_flat_list = [num for sublist in survived for num in sublist]
    df_id['Survived'] = survived_flat_list
    df_id.to_csv(filepath)
    #with open(filepath, "w") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(new_list)

Titanic_training = TitanicPreProcessing(filename='data/titanic/train.csv', plotting=True)
X_train, y_train = Titanic_training.df_return()

Titanic_testing = TitanicPreProcessing(filename='data/titanic/test.csv', plotting=False)
X_test = Titanic_testing.df_return()

# STANDARDIZING THE INPUT SETS

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# CHECKING SHAPES OF THE SETS

# print(f"Shape of train set is {X_train.shape}")
# print(f"Shape of test set is {X_test.shape}")
# print(f"Shape of train label is {y_train.shape}")

# TRAINING THE NEURAL NET

nn = NeuralNet(layers=[8, 4, 1], learning_rate=0.01, iterations=500)
nn.fit(X_train, y_train)
nn_train_prediction = nn.predict(X_train)

# CHECKING ACCURACY OF CLASSIFICATION

nn_training_accuracy = nn.acc(y_train, nn_train_prediction)
print(f'The training accuracy of the neural network is: {nn_training_accuracy}')

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

nn_test_prediction = nn.predict(X_test)

# CLASSIFY WITH KNN
knn = Classification(KNeighborsClassifier())
knn.fit(X_train, y_train)
knn_train_prediction = knn.predict(X_train)

# CHECKING ACCURACY OF CLASSIFICATION

knn_training_accuracy = knn.acc(y_train, knn_train_prediction)
print(f'The training accuracy of the K-nearest neighbours is: {knn_training_accuracy}')

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

test_prediction = knn.predict(X_test)

# CLASSIFY WITH LOGISTIC REGRESSION
regression = Classification(LogisticRegression())
regression.fit(X_train, y_train)
regression_train_prediction = regression.predict(X_train)

# CHECKING ACCURACY OF CLASSIFICATION

regression_training_accuracy = regression.acc(y_train, regression_train_prediction)
print(f'The training accuracy of the Logistic Regression is: {regression_training_accuracy}')

regression_test_prediction = regression.predict(X_test)


create_output_csv('data/titanic/test_prediction.csv', knn.predict(X_test))
