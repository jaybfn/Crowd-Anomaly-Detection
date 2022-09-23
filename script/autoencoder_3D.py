# import libraries
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose

# define the model

class AutoEncoder3D():

    def __init__(self, FILTERS, SIZE,BATCH_SIZE ,ACTIVATION):
        """ SIZE is the image size in pixel and FILTER is the number of filters in Conv network
        """
        self.FILTER = FILTERS
        self.SIZE = SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.ACTIVATION = ACTIVATION

    def autoencoder(self):

        model = Sequential()

        # Spatial encoder layer
        model.add(Conv3D(filters= round(self.FILTER),kernel_size=(11,11,1),strides=(2,2,1),padding='valid',input_shape=(self.SIZE,self.SIZE,self.BATCH_SIZE, 1),activation=self.ACTIVATION))
        model.add(Conv3D(filters= round(self.FILTER/2),kernel_size=(5,5,1),strides=(1,1,1),padding='valid',activation=self.ACTIVATION))
        # temporal encoder layer
        model.add(ConvLSTM2D(filters= round(self.FILTER/2),kernel_size=(3,3),strides=1,padding='same',dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        model.add(ConvLSTM2D(filters = round(self.FILTER/4),kernel_size=(3,3),strides=1,padding='same',dropout=0.1,return_sequences=True))
        model.add(ConvLSTM2D(filters= round(self.FILTER/2),kernel_size=(3,3),strides=1,padding='same',dropout=0.1, return_sequences=True))
        # spatial decoder
        model.add(Conv3DTranspose(filters= round(self.FILTER/2),kernel_size=(5,5,1),strides=(1,1,1),padding='valid',activation=self.ACTIVATION))
        model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(2,2,1),padding='valid',activation=self.ACTIVATION))

        return model

if __name__ == '__main__':

    FILTERS = 512
    SIZE = 127  
    ACTIVATION = 'tanh'
    BATCH_SIZE = 10
    # initialize the model

    AutoEncoder_model = AutoEncoder3D(FILTERS, SIZE, BATCH_SIZE, ACTIVATION)
    model = AutoEncoder_model.autoencoder()
    print(model.summary())