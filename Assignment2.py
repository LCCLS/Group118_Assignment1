import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNet


class TitanicPreProcessing:

    def __init__(self, filename, plotting):
        self.df = pd.DataFrame(pd.read_csv(filename, delimiter=","))
        #self.test_data = pd.DataFrame(pd.read_csv('data/titanic/test.csv', delimiter=","))

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
        columns = ["Pclass", "Sex", "SibSp", "Parch", 'Embarked', 'Age', 'Fare', ]

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
        columns = {'Age': str(30), 'Embarked': str('S')}
        for col in columns.keys():
            self.df[col].fillna(columns[col], inplace=True)

    def df_return(self):
        if 'Survived' in self.df.columns:
            X = self.df.drop(columns=['Survived'])
            y = self.df['Survived'].values.reshape(X.shape[0], 1)
            return X, y
        else:
            return self.df


Titanic_training = TitanicPreProcessing(filename='data/titanic/train.csv', plotting=True)
X_train, y_train = Titanic_training.df_return()

Titanic_testing = TitanicPreProcessing(filename='data/titanic/test.csv', plotting=False)
X_test = Titanic_testing.df_return()

# STANDARDIZING THE INPUT SETS

sc = StandardScaler()
sc.fit(X_train)
Xtrain = sc.transform(X_train)
Xtest = sc.transform(X_test)
print(Xtrain.isna().sum())

# CHECKING SHAPES OF THE SETS

# print(f"Shape of train set is {X_train.shape}")
# print(f"Shape of test set is {X_test.shape}")
# print(f"Shape of train label is {y_train.shape}")

# TRAINING THE NEURAL NET

nn = NeuralNet(layers=[8, 4, 1], learning_rate=0.01, iterations=500)
nn.fit(X_train, y_train)
train_prediction = nn.predict(Xtrain)

# CHECKING ACCURACY OF CLASSIFICATION

training_accuracy = nn.acc(y_train, train_prediction)
print(f'The training accuracy is: {training_accuracy}')

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

test_prediction = nn.predict(Xtest)


