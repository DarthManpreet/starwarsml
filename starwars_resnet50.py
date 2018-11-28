"""

Star Wars ResNet50

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50 
import numpy as np
import json

classpath = './images/Classes/'
datapath = './images/TestData/'

print("Loading training images...")
training_images = ImageDataGenerator()
train_images = training_images.flow_from_directory(directory=classpath, target_size=(224,224), batch_size=32)

model = resnet50.ResNet50(include_top=True,weights=None,classes=train_images.num_classes)

model.compile(loss='mean_squared_error', optimizer='sgd')

i = 0
for i in range(len(train_images)):
   print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
   model.fit(x=train_images[i][0], y=train_images[i][1], epochs=20)

model.save('./resnet.h5')

#write out any remaining doics (or the whole file if THRESHOLD == 0), could be empty
with open('classes.json', 'w') as outfile:
         json.dump(train_images.class_indices, outfile, indent=4)
