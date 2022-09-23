# importing libraries

import numpy as np
import cv2
from PIL import Image as im
from PIL import Image
import os
from os import listdir
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MinMaxScaler

# image to numpy transformation
image_gray = []
def data_preprocessing(image):
    
    img=img_to_array(image)
    img=resize(img,(127,127,3))
    gray_img=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
    gray_img_scaled = scaler.fit_transform(gray_img)
    image_gray.append(gray_img_scaled)
    return image_gray

if __name__ == '__main__':

    # data normalization
    scaler = MinMaxScaler()

    # path to the images
    src_path_train = "../data/4fps/frames_train/"
    src_path_val = "../data/4fps/frames_val/"
    src_path_test = "../data/4fps/frames_test/"

    # image to numpy transformation
    folder_dir = [src_path_train, src_path_val, src_path_test]
    file_names = ['train.npy', 'validation.npy', 'test.npy']
    for dir, names in zip(folder_dir, file_names):
        for images in os.listdir(dir):
            img = Image.open(dir + images)
            image_gray = data_preprocessing(img)
        gray_npimg = np.array(image_gray)
        nr_img, x_size, y_size = gray_npimg.shape
        gray_npimg.resize(x_size, y_size,nr_img)
        gray_npimg=np.clip(gray_npimg,0,1)
        np.save('../data/4fps/'+names, gray_npimg)



