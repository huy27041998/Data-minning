import numpy as np
import cv2
from tensorflow import keras
model = keras.models.load_model('digit_classification_model1.h5')

digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
paths = [
    'E:/Desktop/main/Img/1.png',
    'E:/Desktop/main/Img/2.png',
    'E:/Desktop/main/Img/5.png',
    'E:/Desktop/main/Img/a.png',
    'E:/Desktop/main/Img/b.png',
    'E:/Desktop/main/Img/c.png',
    'E:/Desktop/main/Img/d.png',
    'E:/Desktop/main/Img/e.png',
    'E:/Desktop/main/Img/i.png',
    'E:/Desktop/main/Img/j.png',
    'E:/Desktop/main/Img/l.png',
    'E:/Desktop/main/Img/o.png',
    'E:/Desktop/main/Img/z.png',
]
for path in paths:
    img = cv2.imread(path)
    cv2.waitKey(0)
    try:
        if (img == None):
            print('Img at', path, 'not found')
            continue
    except ValueError:
        pass
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    listRect = [cv2.boundingRect(contours[i]) for i in range(len(contours))]
    listRect.sort(key=lambda x: -x[2] * x[3])
    x, y, w, h = listRect[0]
    img1 = img[y:y+h, x:x+w].copy()
    img1 = cv2.resize(img1, (28, 28), cv2.INTER_LINEAR)
    img1 = img1 / 255
    img1 = np.reshape(img1, (28, 28, 1))
    res = model.predict(np.array([img1]))
    d = np.argmax(res)
    acc = np.amax(res)
    print(path [-5] + ': ' + digit[d] + ', ' + str(int(acc*100))+'%')