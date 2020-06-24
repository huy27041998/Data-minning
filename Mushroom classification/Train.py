from LoadData import LoadData
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report
import matplotlib.pyplot as plt


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
        self.attribute = [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf.fit(self.train_data['data'], self.train_data['class_name'])
        tree.export_graphviz(clf,
                             out_file='tree.dot',
                             feature_names=self.attribute)
        Y_pred = clf.predict(self.test_data['data'])
        self.metric(self.test_data['class_name'], Y_pred, data_type='binary')

    def metric(self, Y_test, Y_pred, data_type=None):
        print(confusion_matrix(Y_test, Y_pred))
        print(classification_report(Y_test, Y_pred, target_names=self.label))

    def metric_prunning(self):
        self.loadData('mushrooms.csv')
        self.train_and_test_split(0.75)
        self.label = ['p', 'e']
        self.attribute = [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
        path = clf.cost_complexity_pruning_path(self.train_data['data'],
                                                self.train_data['class_name'])
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1],
                impurities[:-1],
                marker='o',
                drawstyle="steps-post")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Total impurity of leaves")
        ax.set_title("Total Impurity vs alpha for training set")

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = tree.DecisionTreeClassifier(random_state=0,
                                              ccp_alpha=ccp_alpha)
            clf.fit(self.train_data['data'], self.train_data['class_name'])
            clfs.append(clf)
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()
        train_scores = [
            clf.score(self.train_data['data'], self.train_data['class_name'])
            for clf in clfs
        ]
        test_scores = [
            clf.score(self.test_data['data'], self.test_data['class_name'])
            for clf in clfs
        ]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas,
                train_scores,
                marker='o',
                label="train",
                drawstyle="steps-post")
        ax.plot(ccp_alphas,
                test_scores,
                marker='x',
                label="test",
                drawstyle="steps-post")
        ax.legend()
        plt.show()