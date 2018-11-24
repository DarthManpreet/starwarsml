"""

Star Wars VGG19

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import vgg19
import numpy as np
#import argparse
import cv2
import os

#orig = cv2.imread('./TestData/ro9kcvX.jpg')

classpath = './Classes/'

test_images = []
training_images = []
training_classes = []

print("Loading training images...")      
#load training images 
for dirname, dirnames, filenames in os.walk(classpath):

    for filename in filenames:
        #load an image and prepare it
        image = image_utils.load_img(os.path.join(dirname, filename), target_size=(224,224))
        image = image_utils.img_to_array(image)
        #image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        training_images.append(image)
        #class name associated with a particular image
        training_classes.append(str(dirname.split('/')[-1:][0]))

#training_images = np.array(training_images)
#training_classes = np.array(training_classes)
"""
print(str(training_images))
print(str(training_images.shape))
input()
print(str(training_classes))
print(str(training_classes.shape))
input()
""" 
model = vgg19.VGG19(include_top=True,weights=None,classes=24)

model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(x=training_images, y=training_classes)
