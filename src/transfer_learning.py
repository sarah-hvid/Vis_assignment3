"""
A script that uses transfer learning with a pretrained CNN (VGG16) for feature extraction. 
As input, you may specify the following parameters:
    A name for the plot and classification report using flag -n 
    How many epochs to run using flag -epochs
    The batch size using flag -batch_size
If unspecified, standard values will be used. 
"""

# system tools
import sys, os
import argparse

# tensorflow tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# plotting
import numpy as np
import matplotlib.pyplot as plt


# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-n", "--names", required = False, help = "specify a name for the plot and classification report")
    ap.add_argument("-epochs", "--epoch_num", required = False, help = "number of epochs", type = int)
    ap.add_argument("-batch_size", "--batch_size", required = False, help = "the batch size", type = int)
        
    args = vars(ap.parse_args())
    return args


# plotting function of the history of the training
def plot_history(H, epochs):
    
    args = parse_args()
    name = args['names']
    
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    # specify name for the plot
    if name == None:
        outpath = os.path.join('output', 'plot_history.png')
        plt.savefig(outpath)
        
    else:
        outpath = os.path.join('output', f'{name}.png')
        plt.savefig(outpath)
    
    return plt

    
def load_vgg16():
    # clear the session for previous model runs
    tf.keras.backend.clear_session()

    # load model
    model = VGG16(include_top = False, # remove classifier layer
                  pooling = 'avg', # optimizer
                  input_shape = (32, 32, 3)) # define our new images input shape
    
    # make each layer non trainable
    for layer in model.layers:
        layer.trainable = False
        
    return model
    
    
def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize
    X_train = X_train/255
    X_test = X_test/255
    
    # binarize labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
   
    return X_train, y_train, X_test, y_test


# function specifying the new classifier layer 
def classifier_layer(model):
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu')(flat1)
    output = Dense(10, activation = 'softmax')(class1)

    # define new model
    model = Model(inputs = model.inputs,
                  outputs = output)
    
    return model


# function to compile the new model
def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # the decrease is by an exponential curve
    initial_learning_rate = 0.01, # high learning rate initially
    decay_steps = 10000,
    decay_rate = 0.9) 

    sgd = SGD(learning_rate = lr_schedule)
    
    model.compile(optimizer = sgd,
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    return model


# function to fit the data to the model
def fit_model(model, X_train, y_train, X_test, y_test):
    args = parse_args()
    name = args['names']
   
    epoch_num = args['epoch_num']
    batch_size = args['batch_size']
    
    if epoch_num == None:
        epoch_num = 10
    else:
        pass
    
    if batch_size == None:
        batch_size = 128
    else:
        pass
    
    H = model.fit(X_train, y_train,
              validation_data = (X_test, y_test),
              batch_size = batch_size,
              epochs = epoch_num,
              verbose = 2)
    
    # creating plot
    plot_history(H, epoch_num) # setting epochs

    # creating predictions
    predictions = model.predict(X_test, batch_size=batch_size)
    
    label_names = ['airplane', 'automobile', 'bird', 
                   'cat', 'deer', 'dog', 'frog', 
                   'horse', 'ship', 'truck']
        
    # print and save classification report
    report = classification_report(y_test.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=label_names)
    
    print(report)
    
    if name == None:
        with open("output/report.txt", "w") as f:
            print(report, file=f)
    else:
        with open(f"output/{name}.txt", "w") as f:
            print(report, file=f)
            
    return 


# the process of the script
def main():
    
    # load data
    X_train, y_train, X_test, y_test = load_cifar10()
     
    # model process
    model = load_vgg16()
    model = classifier_layer(model)
    model = compile_model(model)
    fit_model(model, X_train, y_train, X_test, y_test)
    
    print('success')
    return


if __name__ == '__main__':
    main()