from LoadData import LoadData
import numpy as np
from sklearn import tree    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
class Train:
    def __init__(self):
        super().__init__()
    def loadData(self):
        l = LoadData()
        self.data = l.loadData()
    def train_and_test_split(self, percentage):
        self.train_data = {}
        self.test_data = {}
        self.train_data['class_name'], self.test_data['class_name'], self.train_data['data'], self.test_data['data'] = \
            train_test_split(self.data['class_name'], self.data['data'], train_size = percentage)
    def Train(self):
        self.loadData()
        self.train_and_test_split(0.75)
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.train_data['data'], self.train_data['class_name'])
        Y_pred = clf.predict(self.test_data['data'])
        self.metric(self.test_data['class_name'], Y_pred, data_type='binary')
    def metric(self, Y_test, Y_pred, data_type=None):
        print(confusion_matrix(Y_test, Y_pred))
        print(confusion_matrix(Y_test, Y_pred))
        print('Accuracy:', accuracy_score(Y_test, Y_pred))
        if (data_type=='binary'): print('Precision:', precision_score(Y_test, Y_pred))
        if (data_type=='binary'): print('Recall:', recall_score(Y_test, Y_pred))
        if (data_type=='binary'): print('F measure:', 0)