import pandas as pd
import plotly.express as px
from sklearn import preprocessing


class TitanicPreProcessing:

    def __init__(self):
        self.train_data = pd.DataFrame(pd.read_csv('data/titanic/train.csv', delimiter=","))
        self.test_data = pd.DataFrame(pd.read_csv('data/titanic/test.csv', delimiter=","))

        # print(self.train_data.head())
        # print(self.train_data.info())
        # print(self.train_data.isna().sum())

        # BECAUSE THE COLUMN "CABIN" HAS 687 MISSING VALUES WE WILL OMIT IT FROM THE ANALYSIS
        # BECAUSE NAME DOES NOT HAVE A SIGNIFICIANT CORRELATION, IT WILL BE OMITTED TOO
        # TICKET DOES NOT SEEM TO BE A CONSISTENT PREDICTOR FOR SURVIVAL THEREFORE ALSO DELETED

        del self.train_data['Cabin']
        del self.train_data['Name']
        del self.train_data['Ticket']

    def eda_categorical_plotting(self):
        columns = ["Pclass", "Sex", "SibSp", "Parch", 'Embarked', 'Age', 'Fare', ]

        for col in columns:
            fig = px.histogram(self.train_data, x=col, y='Survived', title=col)
            fig.update_layout(bargap=0.2)
            fig.write_image(f"figures/titanic/{col}.png")
            # fig.show()

    def numerical(self):
        columns = ["Sex", 'Embarked']

        for col in columns:
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(self.train_data[col])
            self.train_data[col] = label_encoder.transform(self.train_data[col])

    def nan_replacement(self):
        columns = {'Age': str(30), 'Embarked': str('S')}
        for col in columns.keys():
            self.train_data[col].fillna(columns[col], inplace=True)


filepath = 'data/titanic/train.csv'
tit = TitanicPreProcessing()
tit.nan_replacement()
tit.numerical()
tit.eda_categorical_plotting()
print(tit.train_data.head())

