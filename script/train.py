# import all the necessary libraries

    # calling the model file
from autoencoder_3D import AutoEncoder3D
import glob
import os 
from os import listdir
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image as im
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import random
from datetime import datetime 
from numpy.random import seed

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
plt.style.use('fivethirtyeight')

# function to create all the necessary directory!

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

def img_transformation(generators):
    """ for 3D conv we need an extra dimention in the data"""
    x ,y = generators.__next__()
    x = np.expand_dims(x,axis=4)
    y = x.copy()
    return x ,y

def metricplot(df, xlab, ylab_1,ylab_2, path):
    
    """
    This function plots metric curves and saves it
    to respective folder
    inputs: df : pandas dataframe 
            xlab: x-axis
            ylab_1 : yaxis_1
            ylab_2 : yaxis_2
            path: full path for saving the plot
            """
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = df[xlab], y = df[ylab_1])
    sns.lineplot(x = df[xlab], y = df[ylab_2])
    plt.xlabel('Epochs',fontsize = 12)
    plt.ylabel(ylab_1,fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend([ylab_1,ylab_2], prop={"size":12})
    plt.savefig(path+'/'+ ylab_1)
    #plt.show()


if __name__ == '__main__':
     
    seed(42)
    tf.random.set_seed(42) 
    keras.backend.clear_session()

    #creating main folder
    today = datetime.now()
    today  = today.strftime('%Y_%m_%d')
    path = '../Model_Outputs/'+ today
    create_dir(path)
 
    # creating directory to save model and its output
    EXPERIMENT_NAME = input('Enter new Experiment name:')
    print('\n')
    print('A folder with',EXPERIMENT_NAME,'name has be created to store all the model details!')
    print('\n')
    folder = EXPERIMENT_NAME
    path_main = path + '/'+ folder
    create_dir(path_main)

    # creating directory to save all the metric data
    folder = 'metrics'
    path_metrics = path_main +'/'+ folder
    create_dir(path_metrics)

    # creating folder to save model.h5 file
    folder = 'model'
    path_model = path_main +'/'+ folder
    create_dir(path_model)

    # creating folder to save model.h5 file
    folder = 'model_checkpoint'
    path_checkpoint = path_main +'/'+ folder
    create_dir(path_checkpoint) 

    # path for the image dataset
    src_path_train = "../data/frames_train/"
    src_path_val = "../data/frames_val/"
    src_path_test = "../data/frames_test/"

    #Size of our input images
    SIZE = 127
    # model parameters
    FILTERS = 256
    SIZE = 227
    ACTIVATION = 'relu'
    BATCH_SIZE = 10
    EPOCHS = 3

    # initializing ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)

    # preparing dataset to be accessed by the model directly from the drive
    train_generator = datagen.flow_from_directory(
        src_path_train,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='input',
        color_mode = 'grayscale'
        )

    validation_generator = datagen.flow_from_directory(
        src_path_val,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='input',
        color_mode = 'grayscale'
        )

    test_generator = datagen.flow_from_directory(
        src_path_test,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='input',
        color_mode = 'grayscale'
        )

    # image transformation by expanding the dimention of image as required by the model
    x_train, y_train = img_transformation(train_generator)
    x_val, y_val = img_transformation(validation_generator)
    x_test, y_test = img_transformation(test_generator)

    # load the model
    AutoEncoder_model = AutoEncoder3D(FILTERS, SIZE, ACTIVATION)
    model = AutoEncoder_model.autoencoder()
    print(model.summary())

    # loading weights:
    #model.load_weights(path_model+'/'+model_name)

    initial_learning_rate = 0.00001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    # compling the model
    model.compile(optimizer=keras.optimizers.Adam(lr_schedule), 
                loss='mean_squared_error',metrics=['accuracy','mean_squared_error'])

    cb = [
        tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
        tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=False)]   

    history = model.fit(x_train, y_train,
            batch_size = BATCH_SIZE, 
            epochs = EPOCHS,
            verbose = 1, 
            validation_data = (x_val, y_val),
            callbacks=[cb])

    # save the model

    model.save(path_model+'/'+'model.h5')

    # reading the data.csv where all the epoch training scores are stored
    df = pd.read_csv(path_metrics+'/'+'data.csv')   

    metricplot(df, 'epoch', 'loss','val_loss', path_metrics)
    metricplot(df, 'epoch', 'accuracy','val_accuracy', path_metrics)
    metricplot(df, 'epoch','mean_squared_error', 'val_mean_squared_error', path_metrics)
    
