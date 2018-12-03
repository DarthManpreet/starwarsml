"""

Star Wars image Convelutional Nueral Network experiments
CS545
Fall 2018

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
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

def model_resnet50(class_cnt):
    model = ResNet50(include_top=True,weights=None,classes=class_cnt)
    return model


def train_model(model, train_images, epoch_cnt=30):

    model.compile(loss='mean_squared_error', optimizer='sgd')

    i = 0
    for i in range(len(train_images)):
       print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
       history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, verbose=1)
       #history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, validation_split=0.25, verbose=1)
       #history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, validation_data=
       #                     (train_images[i][0], train_images[i][1]), verbose=1)

    #plot what the model looks like
    plot_model(model, to_file='./' + model.name + '_visualization.png', show_shapes=True)

    #TODO add an option to visual the history or not.       
    # Plot training & validation accuracy values - appears to be missing from the 
    """
    print(str(history.history.keys()))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./' + model.name + '_training_validation_accuracy.pdf')
    """

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('./' + model.name + '_training_validation_loss.pdf')

    #save off the model so it can be loaded later.
    model.save('./' + model.name + '.h5')

def classify_topfive(modeldata, classlabels, imageloc):

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
                #predictions.append( (int('{:.0f}'.format(results[0][c])), labels[str(c)]) )
                predictions.append( ('{:.11f}'.format(results[0][c]), labels[str(c)]) )
            predictions.sort()
            predictions.reverse()
            test_images[filename] = predictions[:5]
            #test_images[filename] = predictions
            #print("For filename: " + filename + "\n\t" + str(predictions[:5]))

    return test_images

def classify_topfive_image(modeldata, classlabels, imageloc):

    model = load_model(modeldata)

    with open(classlabels) as f:
        labels = json.load(f)
    #print(strlabels))
    test_images = {}

    image = image_utils.load_img(imageloc, target_size=(224,224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    results = model.predict(image)

    #print(str(results))

    predictions = []
    for c in range(len(results[0])):
        predictions.append( ('{:.11f}'.format(results[0][c]), labels[str(c)]) )
    predictions.sort()
    predictions.reverse()
    #test_images[imageloc] = predictions[:5]
    #print("For filename: " + filename + "\n\t" + str(predictions[:5]))

    #return test_images

def verify(modeldata, classlabels, imageloc):

    model = load_model(modeldata)

    #should also contain class labels
    with open(classlabels) as f:
        labels = json.load(f)
    #print(str(labels))
    test_images = {}

    for root, dirs, files in os.walk(imageloc):
        for filename in files:
            image = image_utils.load_img(root + filename, target_size=(224,224))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            print(labels[filename])
            input()

            tmplbl = [] 
            for i in labels[filename]:
                tmplbl.append(i[0])
            tmplbl = np.array(tmplbl)
            #tmplbl = to_categorical(tmplbl, num_classes=19, dtype='float32')
            print(str(tmplbl)) 
            print(str(tmplbl.shape))
            #TODO create numpy array here with test classes.
            input()
            
            #keras.utils.np_utils.to_categorical
            results = model.evaluate(x=image, y=tmplbl)
            print(str(results))
            input()

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
    ap.add_argument("-c5", "--classify", required=False, nargs=3, metavar=('modeldata', 'classlabels', 'imageloc'),
                    help="classify images against a model. usage <model data> <classlabels> <image directory>")
    ap.add_argument("-c5i", "--classifyimage", required=False, nargs=3, metavar=('modeldata', 'classlabels', 'imageloc'),
                    help="classify JUST ONE image against a model. usage <model data> <classlabels> <image file>")
    ap.add_argument("-v5", "--verify", required=False, nargs=3, metavar=('modeldata', 'datalabels', 'imageloc'),
                    help="verify images against a model. usage <model data> <datalabels> <image file>")
 
 
    args = vars(ap.parse_args())

    if args['classify'] != None:
        classifications = classify_topfive(args['classify'][0], args['classify'][1], args['classify'][2])
        with open('testdata_results.json', 'w') as outfile:
             json.dump(classifications, outfile, indent=4)

    #classify a single image
    elif args['classifyimage'] != None:
        classification = classify_topfive_image(args['classifyimage'][0], args['classifyimage'][1], args['classifyimage'][2])
        print(str(classification))

    elif args['verify'] != None:
        classification = verify(args['verify'][0], args['verify'][1], args['verify'][2])

    elif args['train'] != None:
        #TODO add the type of VGG, resnet50, or whatever that is to be trained on instead of hard coding it like below
        train_images, label_map = import_class_images(classpath)
    
        model = model_resnet50(train_images.num_classes)
        train_model(model, train_images, 75)

        #write out classes to a file
        with open(model.name + '_classes.json', 'w') as outfile:
             json.dump(label_map, outfile, indent=4)

        model = model_resnet50(train_images.num_classes)
        train_model(model, train_images, 75)

        #write out classes to a file
        with open(model.name + '_classes.json', 'w') as outfile:
             json.dump(label_map, outfile, indent=4)


if __name__ == "__main__":
    main()
