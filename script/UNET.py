""" Function file for UNET Model"""

# Importing all the necessary libraries

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from keras.regularizers import l2

# Defining variables
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
IMAGE_CHANNELS = 3
n_classes = 1

# Defining UNET model: 

def UNET(n_classes, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):

    """ UNET model 
        This function required 4 inputs:
            1. n_classes : Number of classes in the image ( n_classes = 9 in the mosquito dataset)
            2. IMAGE_HEIGHT : height in pixels
            3. IMAGE_WIDTH : width in pixels
            4. IMAGE_CHANNELS : channels corresponds to grey scale image will have 1 channels and for colored images will be 3 channels
            
    """

    # Input for the convolution2D layer:
    inputs =  Input((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    # Contraction path

    # conv layer 2
    conv_layer1 = Conv2D(32, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same')(inputs) 
    conv_layer1 = Dropout(0.1)(conv_layer1) #dropout of 10%
    conv_layer1 = Conv2D(32, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same')(conv_layer1) 
    pool1 = MaxPooling2D((2,2))(conv_layer1) # maxpool of (2,2) kernel

    # conv layer 2
    conv_layer2 =  Conv2D(64, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same')(pool1) 
    conv_layer2 =  Dropout(0.1)(conv_layer2) 
    conv_layer2 =  Conv2D(64, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same')(conv_layer2) 
    pool2 =  MaxPooling2D((2,2))(conv_layer2)

    # conv layer 3
    conv_layer3 =  Conv2D(128, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same' )(pool2) 
    conv_layer3 =  Dropout(0.2)(conv_layer3) 
    conv_layer3 =  Conv2D(128, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same' )(conv_layer3) 
    pool3 =  MaxPooling2D((2,2))(conv_layer3)

    # conv layer 4
    conv_layer4 =  Conv2D(256, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same' )(pool3) 
    conv_layer4 =  Dropout(0.2)(conv_layer4) 
    conv_layer4 =  Conv2D(256, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same' )(conv_layer4) 
    pool4 =  MaxPooling2D((2,2))(conv_layer4)

    # conv layer 5
    conv_layer5 =  Conv2D(512, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same'  )(pool4) 
    conv_layer5 =  Dropout(0.3)(conv_layer5) 
    conv_layer5 =  Conv2D(512, (3,3), activation='relu', kernel_initializer= 'he_normal', padding='same'  )(conv_layer5) 

    #Expansion path

    # conv layer 1
    Trans_conv_layer1 =  Conv2DTranspose(256, (2,2), strides=(2,2), padding='same',  )(conv_layer5) # upscaling
    Trans_conv_layer1 =  concatenate([Trans_conv_layer1, conv_layer4])
    conv_layer6 =  Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same' )(Trans_conv_layer1)
    conv_layer6 =  Dropout(0.2)(conv_layer6)
    conv_layer6 =  Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same' )(conv_layer6)

    # conv layer 2
    Trans_conv_layer2 =  Conv2DTranspose(128, (2,2), strides=(2,2), padding='same',  )(conv_layer6)
    Trans_conv_layer2 =  concatenate([Trans_conv_layer2, conv_layer3])
    conv_layer7 =  Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(Trans_conv_layer2)
    conv_layer7 =  Dropout(0.2)(conv_layer7)
    conv_layer7 =  Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(conv_layer7)

    # conv layer 3
    Trans_conv_layer3 =  Conv2DTranspose(64, (2,2), strides=(2,2), padding='same',  )(conv_layer7)
    Trans_conv_layer3 =  concatenate([Trans_conv_layer3, conv_layer2])
    conv_layer8 =  Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(Trans_conv_layer3)
    conv_layer8 =  Dropout(0.1)(conv_layer8)
    conv_layer8 =  Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(conv_layer8)

    # conv layer 4
    Trans_conv_layer4 =  Conv2DTranspose(32, (2,2), strides=(2,2), padding='same',  )(conv_layer8)
    Trans_conv_layer4 =  concatenate([Trans_conv_layer4, conv_layer1])
    conv_layer9 =  Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(Trans_conv_layer4)
    conv_layer9 =  Dropout(0.1)(conv_layer9)
    conv_layer9 =  Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same'  )(conv_layer9)

    # Model output
    outputs =  Conv2D(n_classes, (1,1), activation='sigmoid')(conv_layer9)

    model = Model(inputs = [inputs], outputs = [outputs])
    
    return model


if __name__ == '__main__':
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    IMAGE_CHANNELS = 1
    n_classes = 1

    mod = UNET(n_classes, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    mod.summary()