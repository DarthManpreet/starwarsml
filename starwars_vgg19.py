"""

Star Wars VGG19

"""

from keras_preprocessing import image as image_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import preprocess_input
from keras_applications import vgg19
from keras_applications import resnet50
import numpy as np
#import argparse
import cv2

#orig = cv2.imread('./TestData/ro9kcvX.jpg')

image = image_utils.load_img('./TestData/ro9kcvX.jpg', target_size=(224,224))

model = vgg19.VGG19(include_top=True,input_shape=(224, 224, 3),weights=None,classes=24)

