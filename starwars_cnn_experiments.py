"""

Star Wars image Convelutional Nueral Network experiments
CS545
Fall 2018

"""

from keras import backend as K

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
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

from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd

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


def model_vgg16(class_cnt):
    model = VGG16(include_top=True,weights=None,classes=class_cnt)
    return model

def model_vgg19(class_cnt):
    model = VGG19(include_top=True,weights=None,classes=class_cnt)
    return model

def model_resnet50(class_cnt):
    model = ResNet50(include_top=True,weights=None,classes=class_cnt)
    return model


def train_model(model, train_images, epoch_cnt=30):

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    i = 0
    for i in range(len(train_images)):
       print("********* Starting Batch " + str(i+1) + " out of " + str(len(train_images)) + "*********")
       #history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, verbose=1)
       history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, validation_split=0.20, verbose=1)
       #history = model.fit(x=train_images[i][0], y=train_images[i][1], epochs=epoch_cnt, validation_data=
       #                     (train_images[i][0], train_images[i][1]), verbose=1)

    #plot what the model looks like
    plot_model(model, to_file='./' + model.name + '_visualization.png', show_shapes=True)

    #TODO add an option to visual the history or not.       
    # Plot training & validation accuracy values - appears to be missing from the 
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('./' + model.name + '_training_validation_accuracy.pdf')

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.legend(['Train'], loc='upper left')
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
    test_images[imageloc] = predictions[:5]
    #print("For filename: " + filename + "\n\t" + str(predictions[:5]))

    return test_images

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
            tmplbl = np.asarray(tmplbl)
            #tmplbl = to_categorical(tmplbl, num_classes=19, dtype='float32')
            print(str(tmplbl))
            print(tmplbl) 
            print(str(tmplbl.shape))
            print(image.shape)

            tmplbl = np.asarray([tmplbl])
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

def accuracy(modeldata, classlabels, datalabels, imageloc):

    model = load_model(modeldata)

    with open(classlabels) as f:
        clabels = json.load(f)
    print(str(clabels))

    #should also contain class labels
    with open(datalabels) as f:
        dlabels = json.load(f)
    #print(str(labels))
    test_images = {}

    total = 0 
    correct = 0
    confmatrix =  [ [ ] ]

    true_label = []
    prediction_label = []

    for root, dirs, files in os.walk(imageloc):
        for filename in files:
            image = image_utils.load_img(root + filename, target_size=(224,224))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            
            results = model.predict(x=image)

            predictions = []
            for c in range(len(results[0])):
                predictions.append( ('{:.11f}'.format(results[0][c]), clabels[str(c)]) )
            predictions.sort()
            predictions.reverse()

            print(str(predictions[0]))
            print(str(dlabels[filename]))

            tmplbl = []
            total = total + 1
            cor = False
            for i in dlabels[filename]:
                if i[1] == predictions[0][1] and i[0] == 1:
                    print(str(i[1]))
                    print(str(predictions[0][1]))
                    print(str(i[0]))
                    correct = correct + 1
                    cor = True

                    true_label.append(i[1])
                    prediction_label.append(predictions[0][1])
            
            if cor is False:
                for i in dlabels[filename]:
                    if i[0] == 1:
                        true_label.append(i[1])
                        prediction_label.append(predictions[0][1])
                        break


            """
            #TODO build a basic confusion matrix for our classes
            if cor == True:
                #TODO confusion matrix
                if i[0] == 1:
                    if i[1] not in confmatrix_actual
            #some of the validation wallpapers have more than one class,
            #if it didn't get any of them correctly, just grab the first
            #one you find that it missed for the confusion matrix.
            else:

            """

            print("Correct: " + str(correct))         
            print("Total: " + str(total)) 

    cm = pd.DataFrame(confusion_matrix(true_label, prediction_label, labels=["ATAT","Ben Kenobi","Boba Fett","C3PO","Chewbacca","Darth Vader","Death Star","Han Solo","Lando Calrissian","Luke Skywalker","Millennium Falcon","Princess Leia","R2D2","Star Destroyer","Storm Trooper","Tie Fighter","Wicket","X-Wing","Yoda"]), index=["ATAT","Ben Kenobi","Boba Fett","C3PO","Chewbacca","Darth Vader","Death Star","Han Solo","Lando Calrissian","Luke Skywalker","Millennium Falcon","Princess Leia","R2D2","Star Destroyer","Storm Trooper","Tie Fighter","Wicket","X-Wing","Yoda"], columns=["ATAT","Ben Kenobi","Boba Fett","C3PO","Chewbacca","Darth Vader","Death Star","Han Solo","Lando Calrissian","Luke Skywalker","Millennium Falcon","Princess Leia","R2D2","Star Destroyer","Storm Trooper","Tie Fighter","Wicket","X-Wing","Yoda"])
    plt.figure(1)
    sb.heatmap(cm, annot=True, fmt="d", cmap="Greens", linewidths=1, linecolor='black')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("exp_cm.png")
    plt.clf()

    print (str(correct / total))
    return test_images



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False, nargs=2, metavar=('modelname', 'epochs'),
                    help="train a model, supply <modelname> and how many <epochs>")
    ap.add_argument("-c5", "--classify", required=False, nargs=3, metavar=('modeldata', 'classlabels', 'imageloc'),
                    help="classify images against a model. usage <model data> <classlabels> <image directory>")
    ap.add_argument("-c5i", "--classifyimage", required=False, nargs=3, metavar=('modeldata', 'classlabels', 'imageloc'),
                    help="classify JUST ONE image against a model. usage <model data> <classlabels> <image file>")
    ap.add_argument("-v5", "--verify", required=False, nargs=3, metavar=('modeldata', 'datalabels', 'imageloc'),
                    help="verify images against a model. usage <model data> <datalabels> <image file>")
    ap.add_argument("-a", "--accuracy", required=False, nargs=4, metavar=('modeldata', 'classlabels', 'datalabels', 'imageloc'),
                    help="check model accuracy. usage <model data> <classlabels> <datalabels> <image file>")
 
 
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

    elif args['accuracy'] != None:
        classification = accuracy(args['accuracy'][0], args['accuracy'][1], args['accuracy'][2], args['accuracy'][3])

    elif args['train'] != None:
        train_images, label_map = import_class_images(classpath)

        if args['train'][0] == 'vgg16':
            print("Training a vgg16 for " + args['train'][1] + " epochs")
            model = model_vgg16(train_images.num_classes)
            train_model(model, train_images, int(args['train'][1]))

            #write out classes to a file
            with open(model.name + '_classes.json', 'w') as outfile:
                 json.dump(label_map, outfile, indent=4)

        elif args['train'][0] == 'vgg19':
            print("Training a vgg19 for " + args['train'][1] + " epochs")
   
            model = model_vgg19(train_images.num_classes)
            train_model(model, train_images, int(args['train'][1]))

            #write out classes to a file
            with open(model.name + '_classes.json', 'w') as outfile:
                 json.dump(label_map, outfile, indent=4)

        elif args['train'][0] == 'resnet50':
            print("Training a resnet50 for " + args['train'][1] + " epochs")
            model = model_resnet50(train_images.num_classes)
            train_model(model, train_images, int(args['train'][1]))

            #write out classes to a file
            with open(model.name + '_classes.json', 'w') as outfile:
                 json.dump(label_map, outfile, indent=4)
    else:
        print("Nothing to do")

if __name__ == "__main__":
    main()
