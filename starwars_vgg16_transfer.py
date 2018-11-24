"""

Star Wars VGG16
Transfer learning attempt

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
import json
import numpy as np

classpath = './Classes/'
datapath = './TestData/'

test_images = []
test_classes = []

print("Loading training images...")

training_images = ImageDataGenerator()
train_images = training_images.flow_from_directory(directory=classpath, target_size=(224,224), batch_size=32)

"""
    for filename in filenames:
        #load an image and prepare it
        image = image_utils.load_img(os.path.join(dirname, filename), target_size=(224,224))
        image = image_utils.img_to_array(image)
        #print(str(image[0][0]))
        #print(str(image.shape))
        #input()
        image = np.expand_dims(image, axis=0)
        #image = preprocess_input(image)
        test_images.append(image)
        #class name associated with a particular image

"""

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

#Create your own input format (here 1920,1080,3)
input = Input(shape=(224,224,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(train_images.num_classes, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd')

i = 0
for i in range(len(train_images)):
   print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
   model.fit(x=train_images[i][0], y=train_images[i][1], epochs=20)
   #model.fit(x=xt, y=yk, epochs=20)

model.save('./vgg19_transfer.h5')

#write out any remaining doics (or the whole file if THRESHOLD == 0), could be empty
with open('classes.json', 'w') as outfile:
         json.dump(train_images.class_indices, outfile, indent=4)

