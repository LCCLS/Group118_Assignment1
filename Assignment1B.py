import pandas as pd


class preprocessing:

    def __init__(self, filepath):
        self.df = self.reader(filepath)
        self.cleaning()

    def __repr__(self):
        return self.df

    @staticmethod
    def reader(filepath):
        """
        reads a csv file
        :param filepath: path to the file
        :return: only reads the columns we want
        """
        input_vars = ['income', 'social_fear', 'improve_yourself_how', 'friends', 'depressed']
        return pd.DataFrame(pd.read_csv(filepath, delimiter=",", usecols=input_vars))

    def cleaning(self):
        """
        Clenas the improve_yourself_how column. it counts the commas in an answer
        :return: the number of self-improvements of a person (none, 1, 2+)
        """
        for answer in self.df['improve_yourself_how']:
            occurrence = answer.count(',')

            if occurrence == 1:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=str('1'))
            elif occurrence >= 2:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=str('2+'))
            else:
                self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                          value=None)

            self.df['improve_yourself_how'] = self.df['improve_yourself_how'].replace(to_replace=answer,
                                                                                      value=int(occurrence))
        self.df.rename(columns={'improve_yourself_how': 'number_of_self_improvements'}, inplace=True)


path = 'data/forever_alone.csv'
preprocessed_file = preprocessing(path)
