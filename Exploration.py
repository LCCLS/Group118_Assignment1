import pprint
import csv
from matplotlib import pyplot as plt
import pandas as pd


class Data_Exploration:

    def __init__(self, filename: str):

        # class vars
        self.ODI = []
        self.ODI_frequencies = {}
        self.questions = []
        self.all_responses = []

        # reading data
        self.csv_reader(filename)

        #preprocessing of data --> needs to be done
        self.preprocessing()

        #getting all answers and their frequencies
        self.response_mapping()
        #pprint.pprint(self.ODI_frequencies)

        #plotting the results
        self.plot('Tijdstempel')

    def csv_reader(self, input_file):

        with open(input_file, "r", encoding="utf-8-sig") as infile:
            csv_reader = csv.reader(infile, delimiter=";")
            for row in csv_reader:
                self.ODI.append(row)

    def preprocessing(self):

        # check what kind of preprocessing is needed and revise this method
        # maybe use spacy to get tokens, handle missing values, etc.

        pass

    def response_mapping(self):

        self.questions = self.ODI[0]
        self.all_responses = self.ODI[1:]
        self.ODI_frequencies = {self.questions[i]: {} for i in range(len(self.questions))}

        for student in self.all_responses:
            for response in enumerate(student):

                resp = response[1]
                index = response[0]

                if resp not in self.ODI_frequencies[self.questions[index]]:
                    self.ODI_frequencies[self.questions[index]][resp] = 1

                elif resp in self.ODI_frequencies[self.questions[index]].keys():
                    self.ODI_frequencies[self.questions[index]][resp] += 1

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
