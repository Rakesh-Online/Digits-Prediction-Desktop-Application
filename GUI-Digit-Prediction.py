#!/usr/bin/env python
# coding: utf-8


import tkinter as tk
from PIL import Image, ImageTk,ImageDraw
import numpy as np
import cv2
import os
import joblib



win = tk.Tk()

w,h = 500,500
fontbutton = "Helvetica 20 bold"
fontlabel = "Helvetica 24 bold"
count = 0

model = joblib.load(r'C:\Users\rakes\Documents\Desktop\Bepec(CT)\Machine Learning\KNN\KNN-Handwritten-Digits.sav')

def eventFunction(event):
    
    x = event.x
    y = event.y
    
    x1 = x-30
    y1 = y-30
    
    x2 = x+30 
    y2 = y+30 
    
    canvas.create_oval((x1,y1,x2,y2), fill = "black")
    img_Draw.ellipse((x1,y1,x2,y2), fill="white")


def save():
    global count
    
    imgarray = np.array(img)
    imgarray = cv2.resize(imgarray,(8,8))
    path=os.path.join('data',str(count)+'.jpg')
    cv2.imwrite(path,imgarray)
    count = count+1
    
def clear():
    
    global img,img_Draw
    
    canvas.delete("all")
    img = Image.new('RGB',(w,h), (0,0,0))
    img_Draw = ImageDraw.Draw(img)
    
    
def predict():
    
    imgarray = np.array(img)
    imgarray = cv2.cvtColor(imgarray,cv2.COLOR_RGB2GRAY)
    imgarray = cv2.resize(imgarray,(8,8))
    imgarray = np.reshape(imgarray,(1,64))
    imgarray = (imgarray/255)*15
    result = model.predict(imgarray)
    labelStatus.config( text= "Predicted Digit : "+ str(result))
    

    
canvas = tk.Canvas(win, width=w,height=h,bg='white')
canvas.grid(row= 0, column=0, columnspan=4)

buttonSave = tk.Button(win, text= "save", bg="green", fg= "white", font= fontbutton, command = save)
buttonSave.grid(row=1,column= 0)

buttonSave = tk.Button(win, text= "predict", bg="blue", fg= "white", font= fontbutton,command = predict )
buttonSave.grid(row=1,column= 1)

buttonSave = tk.Button(win, text= "clear", bg="gray", fg= "white", font= fontbutton,command= clear )
buttonSave.grid(row=1,column= 2)

buttonSave = tk.Button(win, text= "exit", bg="Red", fg= "white", font= fontbutton, command = win.destroy )
buttonSave.grid(row=1,column= 3)

labelStatus = tk.Label(win, text= "Predicted Digit :None", bg= "white", fg="black", font= fontlabel)
labelStatus.grid(row=2,column=0 ,columnspan= 4)

canvas.bind('<B1-Motion>', eventFunction)
img = Image.new('RGB',(w,h), (0,0,0))
img_Draw = ImageDraw.Draw(img)

win.mainloop()






