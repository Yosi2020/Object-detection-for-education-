import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import playsound
import numpy
import time
import cv2
#load the trained model to classify the images
from keras.models import load_model
from efficientnet.tfkeras import EfficientNetB4

model = load_model('./EYUEDU_model_11Class.h5')
#dictionary to label all the Covid-19 detection dataset classes.
classes = { 
    0:'Butterfly',
    1:'Car[Automotive]',
    2:'Cat',
    3:'Chess',
    4:'Chicken',
    5:'Cow',
    6:'Dog',
    7:'Elephant',
    8:'Food',
    9:'Horse',
    10:'Sheep'
}
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Object detection')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path, model):
    global label_packed
    img = Image.open(file_path)
    img = numpy.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float") /255.0
    Y = numpy.expand_dims(img, axis = 0)
    pred = model.predict(Y)
    sign = classes[int(numpy.argmax(pred[0]))]
    print(sign)
    label.configure(foreground='#011638', text=sign)
    Voice_system(sign)
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
                      command=lambda: classify(file_path, model),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
                         font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def Voice_system(name):
    if name == 'Butterfly':
        playsound.playsound('./Voice data/Butterfly.mp3')
        time.sleep(1)
    elif name == 'Car[Automotive]':
        playsound.playsound('./Voice data/Car.mp3')
        time.sleep(1)
    elif name == 'Cat':
        playsound.playsound('./Voice data/Cat.mp3')
        time.sleep(1)
    elif name == 'Chess':
        playsound.playsound('./Voice data/Chess.mp3')
        time.sleep(1)
    elif name == 'Chicken':
        playsound.playsound('./Voice data/Chicken 2.mp3')
        time.sleep(1)
    elif name == 'Cow':
        playsound.playsound('./Voice data/Cow.mp3')
        time.sleep(1)
    elif name == 'Dog':
        playsound.playsound('./voice data/Dog.mp3')
        time.sleep(1)
    elif name == 'Elephant':
        playsound.playsound('./Voice data/Elephant.mp3')
        time.sleep(1)
    elif name == 'Food':
        playsound.playsound('./Voice data/Food.mp3')
        time.sleep(1)
    elif name == 'Horse':
        playsound.playsound('./Voice data/Horse.mp3')
        time.sleep(1)
    elif name == 'Sheep':
        playsound.playsound('./Voice data/Sheep.mp3')
        time.sleep(1)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
                            (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,
              padx=10,pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Object detection",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
