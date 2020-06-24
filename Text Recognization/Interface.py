# from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
from tensorflow import keras
import cv2
model = keras.models.load_model('digit_classification_model2.h5')
np.set_printoptions(suppress=True)
def predict_digit(img):
    #resize image to 28x28 pixels
    img = np.array(img).astype('uint8')
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

    # img = img.resize((28,28))
    # #convert rgb to grayscale
    # img = img.convert('L')
    # img = np.array(img)
    # #reshaping to support our model input and normalizing
    # img = np.reshape(img, (28, 28, 1))
    # img = img/255.0
    #predicting the class
    res = model.predict(np.array([img1]))
    # cv2.imshow('img', img1)
    # cv2.waitKey(0)
    return np.argmax(res), np.amax(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Waiting..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Waiting..")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= self.digit[digit]+', '+ str(int(acc*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()