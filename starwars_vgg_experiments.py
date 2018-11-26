"""

Star Wars VGG experiments
Transfer learning attempt

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import json
import argparse
import os
import numpy as np

classpath = './Classes/'
datapath = './TestData/'



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

def import_class_images(directory):

    print("Loading training images...")
    training_images = ImageDataGenerator()
    train_images = training_images.flow_from_directory(directory=classpath, target_size=(224,224), batch_size=32)

    label_map = (train_images.class_indices)
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v

    return train_images, label_map

#creates a brand new model based on the following parameters
def model_vgg16_transfer(class_cnt):
    
    #attempt to start with imagenet weights and train up from there
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    input = Input(shape=(224,224,3),name = 'image_input')

    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input)

    #Add fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(class_cnt, activation='softmax', name='predictions')(x)

    #Create model 
    model = Model(input=input, output=x)
    model.name = 'vgg16_transfer'

    #model.summary()
    return model

#creates a brand new model based on the following parameters
def model_vgg19_transfer(class_cnt):
    
    #attempt to start with imagenet weights and train up from there
    model_vgg19_conv = VGG19(weights='imagenet', include_top=False)

    input = Input(shape=(224,224,3),name = 'image_input')

    #Use the generated model 
    output_vgg19_conv = model_vgg19_conv(input)

    #Add fully-connected layers 
    x = Flatten(name='flatten')(output_vgg19_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(class_cnt, activation='softmax', name='predictions')(x)

    #Create model 
    model = Model(input=input, output=x)
    model.name = 'vgg19_transfer'

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    #model.summary()
    return model


def model_vgg19(class_cnt):
    model = VGG19(include_top=True,weights=None,classes=class_cnt)
    return model

def train_model(model, train_images, epoch_cnt=30):

    model.compile(loss='mean_squared_error', optimizer='sgd')

    i = 0
    for i in range(len(train_images)):
       print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
       model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt)
       #model.fit(x=xt, y=yk, epochs=20)

    model.save('./' + model.name + '.h5')

def classify(modeldata, classlabels, imageloc):

    model = load_model(modeldata)

    with open(classlabels) as f:
        labels = json.load(f)
    #print(strlabels))
    test_images = {}

    for root, dirs, files in os.walk(imageloc):
        for filename in files:
            image = image_utils.load_img(root + filename, target_size=(224,224))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            results = model.predict(image)

            #print(str(results))

            predictions = []
            for c in range(len(results[0])):
                predictions.append( ('{:.11f}'.format(results[0][c]), labels[str(c)]) )
            predictions.sort()
            predictions.reverse()
            test_images[filename] = predictions[:5]
            #print("For filename: " + filename + "\n\t" + str(predictions[:5]))

    return test_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False,
                    help="train a model")
    ap.add_argument("-c", "--classify", required=False, nargs=3, metavar=('modeldata', 'classlabels', 'imageloc'),
                    help="classify an image against a model. usage <model data> <classlabels> <image directory>")
    args = vars(ap.parse_args())

    if args['classify'] != None:
        classifications = classify(args['classify'][0], args['classify'][1], args['classify'][2])
        with open('testdata_results.json', 'w') as outfile:
             json.dump(classifications, outfile, indent=4)


    elif args['train'] != None:
        #TODO add the type of VGG that is to be trained on instead of hard coding it like below
        train_images, label_map = import_class_images(classpath)
    
        model = model_vgg19_transfer(train_images.num_classes)
        train_model(model, train_images, 75)

        #write out classes to a file
        with open(model.name + '_classes.json', 'w') as outfile:
             json.dump(label_map, outfile, indent=4)


if __name__ == "__main__":
    main()
