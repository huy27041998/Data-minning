import numpy as np
import cv2
new_data = np.loadtxt('X.txt').reshape((72000, 28, 28))
print(new_data.shape)
img = new_data[0]
cv2.imshow('new_data[0]', img)
cv2.waitKey(0)