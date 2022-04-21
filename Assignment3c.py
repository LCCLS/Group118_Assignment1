import pprint

import pandas as pd
import csv
from bs4 import BeautifulSoup
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
from collections import Counter

from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNet


class Sms_Classifier:

    def __init__(self, csv_path):

        self.df = pd.DataFrame
        self.df_spam = pd.DataFrame
        self.df_ham = pd.DataFrame
        self.filepath = csv_path
        self.freq_ham_words = None
        self.freq_spam_words = None

    def clean_html_crap(self):
        """
        fugly function to deal with the html tags and turn them into something readable
        :return: updates the self.df dataframe for further processing
        """

        labels = []
        texts = []
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

        with open(self.filepath) as csv_file:
            file = csv.reader(csv_file, delimiter='\n')

            for row in file:
                string = ''.join(row).lower()
                soup = BeautifulSoup(string, features="html.parser")
                new_soup = soup.text.split(';')

                labels.append(new_soup[0])
                text = new_soup[1].split()
                new_text = " ".join(str(elem) for elem in text)
                texts.append(new_text)

            unzip_file = [{'labels': label, 'texts': text} for label, text in zip(labels, texts)]
            self.df = pd.DataFrame(data=unzip_file[1:])

    def further_preprocessing(self):
        punc = ["(", ")", "-", "[", "]", "{", "}", ";", ":", "'", '"', "<", ">", ".", "..", "...", "/", "?", "@", "#",
                "$", "%", "^", "&", "*", "_", "~", ","]

        more_stops = ['u', "'s", "'m", "n't", "2", "4", "ur", "``", "'ll", "''", ]
        stop_words = [word for word in stopwords.words('english')]
        stop_words = stop_words + more_stops
        # print(stop_words)

        self.df["texts"] = self.df["texts"].apply(word_tokenize)

        self.df["texts"] = self.df['texts'].apply(lambda x: [y for y in ' '.join(x).split() if y not in stop_words])
        self.df["texts"] = self.df['texts'].apply(lambda x: [y for y in ' '.join(x).split() if y not in punc])

    def create_sorted_df(self):
        self.df_spam = self.df[self.df['labels'] == 'spam']
        self.df_ham = self.df[self.df['labels'] == 'ham']

    @staticmethod
    def get_most_frequent_tokens(dataframe):

        word_frequencies = Counter()

        for sentence in dataframe["texts"]:
            words = []
            for token in sentence:
                # Filter out Punctuation
                words.append(token)
            word_frequencies.update(words)

        return word_frequencies.most_common(20)

    def update_df_freq_words(self, common_words, column_name):

        aim_words = [letter for letter, count in common_words]

        word_frequencies = []
        for row in self.df['texts']:
            common = 0
            for word in row:
                if word in aim_words:
                    common += 1
            word_frequencies.append(common)
        # print(word_frequencies, len(word_frequencies))

        self.df[column_name] = word_frequencies

    def get_sentence_length(self):

        sentence_length = []
        for row in self.df['texts']:
            sentence_length.append(len(row))

        self.df['sentence_length'] = sentence_length

    def numerical_y(self):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.df['labels'])
        self.df['labels'] = label_encoder.transform(self.df['labels'])

    def run(self):

        self.clean_html_crap()
        self.further_preprocessing()
        self.create_sorted_df()
        self.freq_ham_words = self.get_most_frequent_tokens(self.df_ham)
        self.freq_spam_words = self.get_most_frequent_tokens(self.df_spam)
        self.update_df_freq_words(self.freq_ham_words, 'freq_ham_words')
        self.update_df_freq_words(self.freq_spam_words, 'freq_spam_words')
        self.get_sentence_length()
        self.numerical_y()

    def df_return(self):

        X = self.df.drop(columns=['labels'])
        X = X.drop(columns=['texts'])
        y = self.df['labels'].values.reshape(X.shape[0], 1)

        return X, y


filepath = "data/SmsCollection.csv"
Sms_spam_or_ham = Sms_Classifier(filepath)
Sms_spam_or_ham.run()
X, y = Sms_spam_or_ham.df_return()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardise the sets
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# TRAINING THE NEURAL NET

nn = NeuralNet(layers=[3, 2, 1], learning_rate=0.001, iterations=500)
nn.fit(X_train, y_train)

# CHECKING ACCURACY OF CLASSIFICATION

nn_train_prediction = nn.predict(X_train)
nn_training_accuracy = nn.acc(y_train, nn_train_prediction)
print(f'The training accuracy of the neural network is: {nn_training_accuracy}')

# MAKING THE PREDICTION OF THE SURVIVAL OF TITANIC PASSENGERS

nn_test_prediction = nn.predict(X_test)
nn_testing_accuracy = nn.acc(y_test, nn_test_prediction)
print(f'The testing accuracy of the neural network is: {nn_testing_accuracy}')

