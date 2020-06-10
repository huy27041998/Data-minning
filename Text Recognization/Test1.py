import cv2
img = cv2.imread('./data/0/train_6f_00000.png', cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('source', img)
print(img.shape)
x, y, w, h = cv2.boundingRect(contours[0])
print(x, y, w, h)
img1 = img[y:y+h, x:x+h].copy()
print(img1.shape)
cv2.imshow('new', img1)
cv2.waitKey(0)
