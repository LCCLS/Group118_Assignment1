from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser


class Data_Exploration:

    def __init__(self, filename: str):

        # class vars

        self.df = pd.DataFrame(pd.read_csv(filename, delimiter=";"))
        self.ODI = []
        self.ODI_frequencies = {}
        self.questions = []
        self.all_responses = []

        # clean data

        self.column_names()
        self.preprocessing_birthdate()
        self.preprocessing_competition_reward()
        self.preprocessing_programme()
        self.preprocessing_good_day_stress_ns()

    def column_names(self):
        """
        changes the lengthy column names to shorter versions.

        :return: a dataframe with updated column names
        """
        column_names = ['timestamp', 'program', 'course_on_ML', 'course_on_IR', 'course_on_stat', 'course_on_databases',
                        'gender', 'chocolate_makes_you', 'birthday', 'nr_neighbor', 'stand_up', 'stress_lev',
                        'competition reward', 'rand_num', 'bed_time_yesterday', 'good_day1', 'good_day2']

        for name in enumerate(self.df.columns):
            self.df.rename(columns={self.df.columns[name[0]]: column_names[name[0]]}, inplace=True)

    def preprocessing_birthdate(self):
        """
        preprocesses the birthdate column. All values are omitted that do not contain DD,MM,YY in some format

        :return: a datetime format of all the birthdates in the column
        """

        for birthdate in self.df['birthday']:
            if 8 <= len(birthdate) < 12 and birthdate != 'NaN':
                try:
                    if 1950 < parser.parse(birthdate).year < 2022:
                        self.df['birthday'] = self.df['birthday'].replace(to_replace=birthdate,
                                                                          value=parser.parse(birthdate))
                    else:
                        self.df['birthday'] = self.df['birthday'].replace(birthdate, 'NaN')
                except ValueError:
                    self.df['birthday'] = self.df['birthday'].replace(birthdate, 'NaN')
            else:
                self.df['birthday'] = self.df['birthday'].replace(birthdate, 'NaN')

    def preprocessing_competition_reward(self):
        """
        preprocesses competition reward. all values that are not numerical are omitted and only values between 0 and 101
        are counted.

        :return:a processes df column of values ranging from 0-100
        """

        for response in self.df['competition reward']:
            if type(response) != str or not response.isnumeric() or not 0 < int(response) < 101:
                self.df['competition reward'] = self.df['competition reward'].replace(response, 'NaN')

    def preprocessing_programme(self):
        """
        preprocesses the programme column and normalizes the names to standards. if the programme count is less than 3,
        the programme is counted to the "others" categorisation

        :return: column of all the programmes of the students
        """

        self.df["program"] = ["AI" if (x.lower() == "ai" or "artificial" in x.lower())
                              else x for x in self.df["program"]]
        self.df["program"] = ["CS" if ("computer" in x.lower() or "computational" in x.lower()) else x for x in
                              self.df["program"]]
        self.df["program"] = ["BA" if ("business" in x.lower()) else x for x in self.df["program"]]
        top3 = ["AI", "BA", "CS"]
        self.df["program"] = ["Other" if ((x not in top3) and (len(self.df[self.df.program == x]) < 3))
                              else x for x in self.df["program"]]

    def preprocessing_yes_no(self):
        """
        preprocesses all the categorical answer options to standardised "0,1,unknown"

        :return: the three columns with numerical values as indicators for responses
        """

        to_be_cleaned = ["course_on_ML", "course_on_stat", "course_on_databases"]
        affirm = ["yes", "ja", "mu"]
        deny = ["no", "sigma", "nee"]
        for col in to_be_cleaned:
            for x in self.df[col]:
                if x in affirm:
                    self.df[col] = self.df[col].replace(x, 1)
                elif x in deny:
                    self.df[col] = self.df[col].replace(x, 0)

    def preprocessing_good_day_stress_ns(self):
        """
        processes the good days columns, the stress column, and the neighbours column.
        Good days columns: all numerical answers are omitted
        stress column: all values not within 0-100 are left out and non-numerical values omitted
        neighbor columns: all non-numerical values are omitted, and values outside of 0-15 are left out

        :return: cleaned df columns in self.df
        """

        clean_up = ["good_day1", "good_day2"]

        self.df["stress_lev"] = [np.nan if (str(x).isnumeric() == False or int(x) > 100 or int(x) < 0) else x for x in
                                 self.df["stress_lev"]]
        self.df["nr_neighbor"] = [np.nan if (str(x).isnumeric() == False or int(x) > 15 or int(x) < 0) else x for x in
                                  self.df["nr_neighbor"]]

        for col in clean_up:
            for x in self.df[col]:
                if x.isnumeric():
                    self.df[col] = self.df[col].replace(x, np.nan)

    # def response_mapping(self):
    #
    #    self.questions = self.ODI[0]
    #    self.all_responses = self.ODI[1:]
    #    self.ODI_frequencies = {self.questions[i]: {} for i in range(len(self.questions))}
    #
    #    for student in self.all_responses:
    #        for response in enumerate(student):
    #
    #            resp = response[1]
    #            index = response[0]
    #
    #            if resp not in self.ODI_frequencies[self.questions[index]]:
    #                self.ODI_frequencies[self.questions[index]][resp] = 1
    #
    #            elif resp in self.ODI_frequencies[self.questions[index]].keys():
    #                self.ODI_frequencies[self.questions[index]][resp] += 1

    def plot(self, dict_key: str):

        # currently only a barchart with faulty x axis categorisation

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()

        df = pd.DataFrame({dict_key: self.ODI_frequencies[dict_key]})
        df[dict_key].value_counts().plot(ax=ax, kind='bar', xlabel=f'{dict_key}', ylabel='frequency')

        plt.show()


document_name = "data/ODI-2022.csv"
exploration = Data_Exploration(filename=document_name)
