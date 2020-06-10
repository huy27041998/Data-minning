import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
class LoadData:
    def __init__(self):
        super().__init__()
    
    def readImage(self, path):
        image = Image.open(path).convert('L')
        array = np.array(image.resize((28, 28)), dtype=np.uint8)
        return array.reshape(28, 28) 
    
    def loadDataImg(self, path):
        print('Reading data:')
        dir_images = os.listdir(path)
        dir_images.sort(key=lambda x: x[0])
        X = np.empty((0, 28, 28))
        Y = np.empty((0, 36))
        for index in tqdm(range(len(dir_images))):
            dir_image = dir_images[index]
            dir_names = os.listdir(path + '/' + dir_image)
            for dir_name in dir_names:
                img = self.readImage(path + '/' + dir_image + '/' + dir_name)
                X = np.append(X, [img], axis=0)
                temp = np.zeros((1, 36))
                temp[0, index] = 1
                Y= np.append(Y, temp, axis=0)
        np.savetxt('Y.txt', Y, fmt='%d')
        with open('X.txt', 'w') as f:
            for x in X:
                np.savetxt(f, x, fmt='%d')
        return {'class_name': Y, 'data': X}
    
    def loadDataFeature(self):
        start = time.time()
        X = np.loadtxt('X_feature.txt')
        Y = np.loadtxt('Y.txt')
        print('load data time: %is' %(time.time() - start))
        print('Train shape: {}'.format(X.shape))
        print('Test shape: {}'.format(Y.shape))
        return {'class_name': Y, 'data' : X}
    
    def loadDataTxt(self):
        start = time.time()
        X = np.loadtxt('X0.txt')
        X = X / 255
        X = np.reshape(X, (72000, 28, 28, 1))
        Y = np.loadtxt('Y.txt')
        print('load data time: %is' %(time.time() - start))
        print('Train shape: {}'.format(X.shape))
        print('Test shape: {}'.format(Y.shape))
        return {'class_name': Y, 'data' : X}

    def loadDataNormalize(self):
        start = time.time()
        X = np.loadtxt('X_normalize.txt').reshape((72000, 28, 28, 1))
        Y = np.loadtxt('Y.txt')
        print('load data time: %i s' %(time.time() - start))
        print('Train shape: {}'.format(X.shape))
        print('Test shape: {}'.format(Y.shape))
        return {'class_name': Y, 'data' : X}

    def handleData(self, data):
        return data