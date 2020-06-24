import numpy as np
import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.metrics import classification_report
from LoadData import LoadData


class Train:
    def __init__(self):
        super().__init__()

    def loadData(self, train_size):
        l = LoadData()
        self.data = l.loadData(train_size)

    def loadDataCSV(self, filename):
        l = LoadData()
        self.data = l.loadDataCSV(filename)

    def train_and_test_split(self, percentage):
        self.train_data = {}
        self.test_data = {}
        self.train_data['class_name'], self.test_data['class_name'], self.train_data['data'], self.test_data['data'] = \
            train_test_split(self.data['class_name'], self.data['data'], train_size = percentage)

    def train_1_class(self):
        l = LoadData()
        stopWords = l.loadStopWords()
        train_sizes = [100, 200]  # size per class
        for train_size in train_sizes:
            print('Training size:',
                  math.floor(train_size * 0.75) * 2, 'Test size:',
                  math.ceil(train_size * 0.25) * 2)
            self.loadData(train_size)
            vect = TfidfVectorizer(stop_words=stopWords)

            # balance classes
            temp_class = self.data['class_name'][train_size:]
            temp_data = self.data['data'][train_size:]
            idx = random.choices(range(len(temp_class)), k=train_size)
            temp_class = [temp_class[i] for i in idx]
            temp_data = [temp_data[i] for i in idx]
            del self.data['data'][train_size:]
            del self.data['class_name'][train_size:]
            self.data['class_name'].extend(temp_class)
            self.data['data'].extend(temp_data)

            self.train_and_test_split(0.75)
            X_train = vect.fit_transform(self.train_data['data'])
            Y_train = [
                1 if i == 'business' else 0
                for i in self.train_data['class_name']
            ]
            X_test = vect.transform(self.test_data['data'])
            Y_test = [
                1 if i == 'business' else 0
                for i in self.test_data['class_name']
            ]
            nb = MultinomialNB()
            Y_pred = nb.fit(X_train, Y_train).predict(X_test)
            self.metric(Y_test, Y_pred)
            print('---------------------------------------------------')

    def metric(self, Y_test, Y_pred, data_type=None):
        print(confusion_matrix(Y_test, Y_pred))
        # print('Accuracy:', accuracy_score(Y_test, Y_pred))
        # if (data_type=='binary'): print('Precision:', precision_score(Y_test, Y_pred))
        # if (data_type=='binary'): print('Recall:', recall_score(Y_test, Y_pred))
        # if (data_type=='binary'): print('F measure:', 0)
        print(classification_report(Y_test, Y_pred))

    def train_n_classes(self):
        l = LoadData()
        stopWords = l.loadStopWords()
        train_sizes = [100, 200, 300]  # size per class
        for train_size in train_sizes:
            print('Training size:',
                  math.floor(train_size * 0.75) * 5, 'Test size:',
                  math.ceil(train_size * 0.25) * 5)
            self.loadData(train_size)
            vect = TfidfVectorizer(stop_words=stopWords)
            self.train_and_test_split(0.75)
            classes = {}
            x = 0
            for i in self.data['class_name']:
                if i not in classes:
                    classes[i] = x
                    x += 1
            X_train = vect.fit_transform(self.train_data['data'])
            Y_train = [classes[i] for i in self.train_data['class_name']]
            X_test = vect.transform(self.test_data['data'])
            Y_test = [classes[i] for i in self.test_data['class_name']]
            nb = MultinomialNB()
            Y_pred = nb.fit(X_train, Y_train).predict(X_test)
            self.metric(Y_test, Y_pred)
            print('---------------------------------------------------')

    def train(self):
        l = LoadData()
        stopWords = l.loadStopWords()
        self.loadDataCSV('bbc-text.csv')
        vect = TfidfVectorizer(stop_words=stopWords)
        self.train_and_test_split(0.75)
        X_train = vect.fit_transform(self.train_data['data'])
        Y_train = self.train_data['class_name']
        X_test = vect.transform(self.test_data['data'])
        Y_test = self.test_data['class_name']
        nb = MultinomialNB()
        Y_pred = nb.fit(X_train, Y_train).predict(X_test)
        self.metric(Y_test, Y_pred)
