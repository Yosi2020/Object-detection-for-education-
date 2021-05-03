#befor start please install pip install git+https://github.com/qubvel/segmentation_models

import cv2
import os
import playsound
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from efficientnet.tfkeras import EfficientNetB4

categories = ['Butterfly', 'Car[Automotive]', 'Cat', 'Chess', 'Chicken', 'Cow',
              'Dog','Elephant', 'Food', 'Horse', 'Sheep']

model = tf.keras.models.load_model("E:/My Project/for education object detection/EYUEDU_model_11Class.h5")

# here please enter the image
file_path = ""

def data_prepare(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Image_size, Image_size))
    img = img.astype("float") /255.0
    img = img_to_array(img)
    image = np.expand_dims(img, axis = 0)
    return image

prediction = model.predict([data_prepare(file_path)])
predict_class = np.argmax(prediction, axis = -1)

name = categories[int(np.argmax(prediction[0]))]

img = cv2.imread(file_path)
new_img = cv2.resize(img, (224, 224))
# Using cv2.putText() method 
image = cv2.putText(new_img, name , (25,25), cv2.FONT_HERSHEY_SIMPLEX ,
                    1, (255,0,0), 2, cv2.LINE_AA)
def Voice_system(name):
    if predict_class == 'Butterfly':
        playsound.playsound(' ')
        time.sleep(1)
    elif predict_class == 'Car[Automotive]':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Cat':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Chess':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Chicken':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Cow':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Dog':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Elephant':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Food':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Horse':
        playsound.playsound('')
        time.sleep(1)
    elif predict_class == 'Sheep':
        playsound.playsound('')
        time.sleep(1)
voice_system(predict_class)
plt.show(image)
plt.show()


