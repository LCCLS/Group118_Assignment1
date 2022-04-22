import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
            return X_train, X_val, y_train, y_val
        else:
            return self.df


def create_output_csv(filepath, survived: list):
    df_id = pd.DataFrame(pd.read_csv('data/titanic/test.csv', delimiter=',', usecols=['PassengerId']))
    survived_flat_list = [num for sublist in survived for num in sublist]
    df_id['Survived'] = survived_flat_list
    df_id.to_csv(filepath)


Titanic_training = TitanicPreProcessing(filename='data/titanic/train.csv', plotting=True)
X_train, X_val, y_train, y_val = Titanic_training.df_return()

Titanic_testing = TitanicPreProcessing(filename='data/titanic/test.csv', plotting=False)
X_test = Titanic_testing.df_return()

# STANDARDIZING THE INPUT SETS

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

# CHECKING SHAPES OF THE SETS

# print(f"Shape of train set is {X_train.shape}")
# print(f"Shape of test set is {X_test.shape}")
# print(f"Shape of train label is {y_train.shape}")

# TRAINING THE NEURAL NET

nn = NeuralNet(layers=[8, 4, 1], learning_rate=0.01, iterations=500)
nn.fit(X_train, y_train)
nn_train_prediction = nn.predict(X_train)
nn_val_prediction = nn.predict(X_val)

# CHECKING ACCURACY OF CLASSIFICATION

nn_training_accuracy = nn.acc(y_train, nn_train_prediction)
nn_val_accuracy = nn.acc(y_val, nn_val_prediction)
print(f'The training accuracy of the neural network is: {nn_training_accuracy}')
print(f'The validation accuracy of the neural network is: {nn_val_accuracy} \n')

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

nn_test_prediction = nn.predict(X_test)

# CLASSIFY WITH KNN
knn = Classification(KNeighborsClassifier(n_neighbors=20))
knn.fit(X_train, y_train)
knn_train_prediction = knn.predict(X_train)
knn_val_prediction = knn.predict(X_val)

# CHECKING ACCURACY OF CLASSIFICATION

knn_training_accuracy = knn.acc(y_train, knn_train_prediction)
knn_val_accuracy = knn.acc(y_val, knn_val_prediction)
print(f'The training accuracy of the K-nearest neighbours is: {knn_training_accuracy}')
print(f'The validation accuracy of the K-nearest neighbours is: {knn_val_accuracy} \n')

# CLASSIFY WITH LOGISTIC REGRESSION
regression = Classification(LogisticRegression())
regression.fit(X_train, y_train)
regression_train_prediction = regression.predict(X_train)
regression_val_prediction = regression.predict(X_val)

# CHECKING ACCURACY OF CLASSIFICATION

regression_training_accuracy = regression.acc(y_train, regression_train_prediction)
regression_val_accuracy = regression.acc(y_val, regression_val_prediction)
print(f'The training accuracy of the Logistic Regression is: {regression_training_accuracy}')
print(f'The validation accuracy of the Logistic Regression is: {regression_val_accuracy} \n')

regression_test_prediction = regression.predict(X_test)

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

knn_test_prediction = knn.predict(X_test)
create_output_csv('data/titanic/test_prediction.csv', knn_test_prediction)
