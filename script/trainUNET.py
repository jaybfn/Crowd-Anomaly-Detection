 #import all the necessary libraries

# calling the model file
#from autoencoder_3D import AutoEncoder3D
from types import new_class
from UNET import UNET
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
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
plt.style.use('fivethirtyeight')

# function to create all the necessary directory!

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

# def img_transformation(generators):
#     """ for 3D conv we need an extra dimention in the data"""
#     x ,y = generators.__next__()
#     x = np.expand_dims(x,axis=4)
#     y = x.copy()
#     return x ,y

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
    src_path_train = "../data/4fps/frames_train/"
    src_path_val = "../data/4fps/frames_val/"
    src_path_test = "../data/4fps/frames_test/"

    # image_size 
    SIZE = 128
    # model parameters
    FILTERS = 128
    ACTIVATION = 'tanh'
    BATCH_SIZE = 20
    EPOCHS = 200
    n_classes = 1
    IMAGE_CHANNELS  = 1
    IMAGE_WIDTH = SIZE
    IMAGE_HEIGHT = SIZE
   

    # model name
    model_name = 'model.h5'

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

    # defining the loss function:
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1*focal_loss)

    metrics = [sm.metrics.IOUScore(threshold = 0.5), sm.metrics.FScore(threshold = 0.5),'accuracy', 'Recall', 'Precision']

    # load the model
    model = UNET(n_classes, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    print(model.summary())


    
    #model_name = 'model.h5'
    # loading weights:

    #model.load_weights(path_model+'/'+model_name)

    initial_learning_rate = 0.00001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    # model compiling:
    model.compile(optimizer = Adam(learning_rate = lr_schedule),
                    loss = total_loss, 
                    metrics = metrics)

    cb = [
        tf.keras.callbacks.ModelCheckpoint(path_model+'/'+model_name),
        tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
        tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=False)]   

    history = model.fit(train_generator,
            batch_size = BATCH_SIZE, 
            epochs = EPOCHS,
            verbose = 1, 
            validation_data = validation_generator,
            callbacks=[cb])

    # save the model

    model.save(path_model+'/'+'model.h5')

    # evaluating validation and testing dataset

    print("____________________________________________________________")
    print("____________________________________________________________")
    print("____________________________________________________________")
    train_IoU = model.evaluate(train_generator,
                                    batch_size = BATCH_SIZE)
    print("Train IoU is = ", (train_IoU[1] * 100.0), "%")

    val_IoU = model.evaluate(validation_generator,
                                    batch_size = BATCH_SIZE)
    print("Val IoU is = ", (val_IoU[1] * 100.0), "%")

    print("____________________________________________________________")
    print("____________________________________________________________")
    print("____________________________________________________________")

    # # calculating losses!

    # train_loss, train_acc, train_MSE = model.evaluate(x_train, y_train)
    # print('\n','Evaluation of Training dataset:','\n''\n','train_loss:',round(train_loss,3),'\n','train_acc:',round(train_acc,3),'\n', 'train_MSE:',round(train_MSE,3))
    
    # val_loss, val_acc, val_MSE = model.evaluate(x_val, y_val)
    # print('\n','Evaluation of Validation dataset:','\n''\n','val_loss:',round(val_loss,3),'\n','val_acc:',round(val_acc,3),'\n', 'val_MSE:',round(val_MSE,3))

    # test_loss, test_acc, test_MSE = model.evaluate(x_test, y_test)
    # print('\n','Evaluation of Testing dataset:','\n''\n','test_loss:',round(test_loss,3),'\n','test_acc:',round(test_acc,3),'\n', 'test_MSE:',round(test_MSE,3))

    # reading the data.csv where all the epoch training scores are stored
    df = pd.read_csv(path_metrics+'/'+'data.csv')   


    metricplot(df,'epoch', 'accuracy', 'val_accuracy', path_metrics)
    metricplot(df,'epoch', 'iou_score', 'val_iou_score', path_metrics)
    metricplot(df,'epoch', 'loss', 'val_loss', path_metrics)
    metricplot(df,'epoch', 'precision', 'val_precision', path_metrics)
    metricplot(df,'epoch', 'recall', 'val_recall', path_metrics)