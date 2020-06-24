from LoadData import LoadData
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Flatten
from tensorflow.compat.v1.keras.layers import MaxPooling2D
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time


class Train:
    def __init__(self):
        super().__init__()
        # tensorflow configuration
        config = tf.ConfigProto(device_count={
            'GPU': 1,
            'CPU': 1
        },
                                log_device_placement=False)
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

    def loadDataTxt(self):
        l = LoadData()
        self.data = l.loadDataTxt()

    def loadDataFeature(self):
        l = LoadData()
        self.data = l.loadDataFeature()

    def loadDataNormalize(self):
        l = LoadData()
        self.data = l.loadDataNormalize()

    def train_and_test_split(self, percentage):
        self.train_data = {}
        self.test_data = {}
        self.train_data['class_name'], self.test_data['class_name'], self.train_data['data'], self.test_data['data'] = \
            train_test_split(self.data['class_name'], self.data['data'], train_size = percentage)

    def Train(self):
        # self.loadDataFeature()
        self.loadDataTxt()
        self.train_and_test_split(0.75)
        # model
        model = Sequential()

        # model.add(Dense(392, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(36, activation='softmax'))

        #cnn model

        model.add(
            Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(36, activation='softmax'))

        # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(
            self.train_data['data'],
            self.train_data['class_name'],
            batch_size=25,
            epochs=100,
            verbose=1,
            validation_data=(self.test_data['data'],
                             self.test_data['class_name']),
        )
        self.model = model
        model.save('digit_classification_model1.h5')
        # Y_pred = model.predict(self.test_data['data'])
        # self.metric(self.test_data['class_name'], Y_pred, data_type='binary')
        self.metric()

    def metric(self):
        self.digit = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        y_true = [np.argmax(i) for i in self.test_data['class_name']]
        y_pred = [
            np.argmax(im) for im in self.model.predict(self.test_data['data'])
        ]
        print(classification_report(y_true, y_pred, target_names=self.digit))
