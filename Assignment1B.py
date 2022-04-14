import pandas as pd


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


path = 'data/forever_alone.csv'
preprocessed_file = preprocessing(path)
preprocessed_file.cleaning_improvement()
preprocessed_file.cleaning_yes_no('depressed')
preprocessed_file.cleaning_yes_no('social_fear')
preprocessed_file.cleaning_income()

print(preprocessed_file.df.head())
print(preprocessed_file.df.shape)
print(preprocessed_file.df.isna().sum())
print(preprocessed_file.df.dtypes)
