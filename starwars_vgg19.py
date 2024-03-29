"""

Star Wars VGG19
This will be for training only, a different file will handle prediction

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg19
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

model = vgg19.VGG19(include_top=True,weights=None,classes=train_images.num_classes)

model.compile(loss='mean_squared_error', optimizer='sgd')

i = 0
for i in range(len(train_images)):
   print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
   model.fit(x=train_images[i][0], y=train_images[i][1], epochs=20)
   #model.fit(x=xt, y=yk, epochs=20)

model.save('./vgg19.h5')

#write out any remaining doics (or the whole file if THRESHOLD == 0), could be empty
with open('classes.json', 'w') as outfile:
         json.dump(train_images.class_indices, outfile, indent=4)

