from LoadData import LoadData
import numpy as np
from sklearn import tree    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report
class Train:
    def __init__(self):
        super().__init__()
    def loadData(self, filename):
        l = LoadData()
        self.data = l.loadData(filename)
    def train_and_test_split(self, percentage):
        self.train_data = {}
        self.test_data = {}
        self.train_data['class_name'], self.test_data['class_name'], self.train_data['data'], self.test_data['data'] = \
            train_test_split(self.data['class_name'], self.data['data'], train_size = percentage)
    def Train(self):
        self.loadData('mushrooms.csv')
        self.train_and_test_split(0.75)
        self.label = ['p', 'e']
        self.attribute = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf.fit(self.train_data['data'], self.train_data['class_name'])
        tree.export_graphviz(clf, out_file='tree.dot', feature_names=self.attribute)    
        Y_pred = clf.predict(self.test_data['data'])
        self.metric(self.test_data['class_name'], Y_pred, data_type='binary')
    def metric(self, Y_test, Y_pred, data_type=None):
        print(confusion_matrix(Y_test, Y_pred))
        print(classification_report(Y_test, Y_pred, target_names=self.label))