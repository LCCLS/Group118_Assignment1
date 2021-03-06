from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
import plotly.express as px
import time


class Data_Exploration:

    def __init__(self, filename: str):

        # class vars

        self.df = pd.DataFrame(pd.read_csv(filename, delimiter=";"))
        self.questions = self.df.columns.values.tolist()
        self.linked_questions = {}

        # clean data

        self.column_names()
        self.preprocessing_birthdate()
        self.preprocessing_competition_reward()
        self.preprocessing_program()
        self.preprocessing_good_day_stress_ns()
        self.preprocessing_bed_time_yesterday()

        # plot data

        self.pie_plotting('program')
        self.pie_plotting('course_on_ML')
        self.pie_plotting('course_on_IR')
        self.pie_plotting('course_on_stat')
        self.pie_plotting('course_on_databases')
        self.pie_plotting('gender')
        self.pie_plotting('chocolate_makes_you')

        self.hist_plotting('number of neighbors')
        self.hist_plotting('birthday')
        self.hist_plotting('bed time yesterday')

        # word clouds

        self.wordcloud('good_day1', second_column='good_day2')

    def column_names(self):
        """
        changes the lengthy column names to shorter versions. also updates self.linked_questions which maps new column
        names to the original questions

        :return: a dataframe with updated column names
        """

        column_names = ['timestamp', 'program', 'course_on_ML', 'course_on_IR', 'course_on_stat', 'course_on_databases',
                        'gender', 'chocolate_makes_you', 'birthday', 'number of neighbors', 'stand up', 'stress_lev',
                        'competition reward', 'rand_num', 'bed time yesterday', 'good_day1', 'good_day2']

        self.linked_questions = {column_names[i]: self.questions[i] for i in range(len(column_names))}

        for name in enumerate(self.df.columns):
            self.df.rename(columns={self.df.columns[name[0]]: column_names[name[0]]}, inplace=True)

    def preprocessing_bed_time_yesterday(self):

        for time in self.df['bed time yesterday']:

            if type(time) != str or not any(char.isdigit() for char in time) or \
                    len([x for x in time if x.isalpha()]) > 2:
                self.df['bed time yesterday'] = self.df['bed time yesterday'].replace(time, np.nan)

            else:
                new_time = time.strip().replace('.', ':').upper()
                if len(new_time) == 1:
                    new_time = f'0{new_time}:00'
                elif len(new_time) == 2:
                    new_time += ':00'
                if new_time[:2].isnumeric() and time.startswith('1') and int(new_time[:2]) < 13:
                    new_time = str(int(new_time[:2]) + 12) + str(new_time[2:])
                if new_time[3:5] != '00':
                    new_time = new_time[:3] + '00'

                try:
                    parts = parser.parse(new_time, ignoretz=True)
                    processed_time = datetime.strftime(parts, '%H:%M:%S')
                    self.df['bed time yesterday'] = self.df['bed time yesterday'].replace(to_replace=time,
                                                                                          value=processed_time)

                except ValueError:
                    self.df['bed time yesterday'] = self.df['bed time yesterday'].replace(time, np.nan)

    # df[df["age"] > 20]

    def preprocessing_birthdate(self):
        """
        preprocesses the birthdate column. All values are omitted that do not contain DD,MM,YY in some format

        :return: a datetime format of all the birthdates in the column
        """

        for birthdate in self.df['birthday']:
            if 8 <= len(birthdate) < 12 and birthdate != 'NaN':
                try:
                    if 1960 < parser.parse(birthdate).year < 2007:
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

    def preprocessing_program(self):
        # this needs to be updated a bit :)
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
                    self.df[col] = self.df[col].replace(x, 'yes')
                elif x in deny:
                    self.df[col] = self.df[col].replace(x, 'no')

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
        self.df["number of neighbors"] = [np.nan if (str(x).isnumeric() == False or int(x) > 15 or int(x) < 0) else '9+'
        if int(x) >= 9 else x for x in self.df["number of neighbors"]]

        for col in clean_up:
            for x in self.df[col]:
                x = str(x.lower())
                if x.isnumeric():
                    self.df[col] = self.df[col].replace(x, 'NaN')

    def pie_plotting(self, column_name):
        """
        takes a column name of the df and creates a plotly pie chart
        :param column_name: the column name of the pd dataframe
        :return: pie chart and saved file in "figures" directory
        """

        df_val = [i for i in self.df[f'{column_name}'].value_counts()]
        df_names = self.df[f'{column_name}'].value_counts().index.to_list()

        fig = px.pie(self.df, values=df_val, names=df_names, title=self.linked_questions[f'{column_name}'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.write_image(f"figures/{self.linked_questions[f'{column_name}']}.png")
        # fig.show()

    def hist_plotting(self, column_name):
        """
        takes a column name of the df and creates a plotly histogram
        :param column_name: the column name of the processed pd dataframe
        :return: histogram and saved file in "figures" directory
        """
        # for i in self.df[f'{column_name}']:
        #    print(i,type(i))

        # the sorting of the values still needs to be changed cause theyre messed up in nr of neighbors

        df_freq = [i for i in self.df[f'{column_name}'].value_counts()]
        df_val = self.df[f'{column_name}'].value_counts().index.to_list()

        df_combined = zip(df_freq, df_val)
        df_combined = sorted(df_combined, key=lambda x: str(x[1:]))
        df_frequencies, df_values = zip(*df_combined)

        fig = px.histogram(self.df, x=df_values, y=df_frequencies, title=self.linked_questions[f'{column_name}'],
                           labels=dict(x=f'{column_name}', y='frequency'))
        fig.update_layout(bargap=0.2)
        fig.write_image(f"figures/{self.linked_questions[f'{column_name}']}.png")
        fig.show()

    def wordcloud(self, column_name, second_column=None):

        # this needs a bit of preprocessing. either we add stopwords manually, or we could use spacy for NLP
        # tokenization.

        """
        takes a column name of the dataframe and plots a wordlcoud
        :param second_column: an additional parameter if a df column name is passed in, the method will concatenate the
                two columns; normally set to default "None"
        :param column_name: df column name
        :return: wordcloud and saved file in "figures" directory
        """

        if second_column:

            all_words = []
            for col in [f'{column_name}', f'{second_column}']:
                for i in self.df[f'{col}']:
                    all_words.append(i)

        else:
            all_words = self.df[f'{column_name}']

        # print(all_words)

        text = "\n".join(str(word).lower() for word in all_words)

        stopwords = set(STOPWORDS)
        stopwords.update(['nan', 'NaN', 'Nan', 'NAN'])

        wordcloud = WordCloud(width=1000, height=600,
                              background_color='black',
                              stopwords=stopwords,
                              min_font_size=10).generate(text)

        plt.figure(figsize=(10, 6), facecolor='black')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=2)
        plt.title(self.linked_questions[f'{column_name}'])
        plt.savefig(f"figures/{self.linked_questions[f'{column_name}']}")
        # plt.show()


document_name = "data/ODI-2022.csv"
exploration = Data_Exploration(filename=document_name)
