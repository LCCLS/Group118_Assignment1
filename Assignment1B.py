import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNet


class PreProcessing:

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
        Clean the improve_yourself_how column. it counts the commas in an answer
        :return: the number of self-improvements of a person (none, 1, 2+)
        """
        self.df['improve_yourself_how'] = self.df['improve_yourself_how'].apply(lambda x: x.split(','))
        self.df['improve_yourself_how'] = self.df['improve_yourself_how'].apply(lambda x: len(x))

        self.df.rename(columns={'improve_yourself_how': 'number_of_self_improvements'}, inplace=True)

    def cleaning_yes_no(self, column_name):
        """
        cleans the column by converting categorical data into binary
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
        max_incomes = []

        for i in self.df['income']:
            income = re.findall(r'\$\d+(?:\,\d+)?', i)
            income = [int(x.replace('$', '').replace(',', '')) for x in income]

            if len(income) == 2:
                max_incomes.append(income[1])
            else:
                max_incomes.append(income[0])

        self.df['income'] = max_incomes


def cross_validation(k, model):
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    sc = StandardScaler()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        acc = model.acc(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / k

    print('Accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy of the NeuralNet: {}'.format(avg_acc_score))


path = 'data/forever_alone.csv'
preprocessed_file = PreProcessing(path)
preprocessed_file.cleaning_improvement()
preprocessed_file.cleaning_yes_no('depressed')
preprocessed_file.cleaning_yes_no('social_fear')
preprocessed_file.cleaning_income()

# CHECKING IF THERES NO MISSING VALUES ALL TYPES ARE THE SAME, ETC.

# print(preprocessed_file.df.head())
# print(preprocessed_file.df.shape)
# print(preprocessed_file.df.isna().sum())
# print(preprocessed_file.df.dtypes)

# FORMATTING DF INTO X AND Y VALUES

X = preprocessed_file.df.drop(columns=['depressed'])
y = preprocessed_file.df['depressed'].values.reshape(X.shape[0], 1)

# STANDARDIZING THE VALUES FOR THE NN

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# CROSS VALIDATION

cross_validation(k=10, model=NeuralNet(layers=[4, 2, 1], learning_rate=0.01, iterations=500))
