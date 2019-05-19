import os
import cv2
from random import shuffle
import glob
import keras
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
K.set_image_data_format('channels_last')

inpdirs = ['./ColorFrames%d'%(i) for i in range(1,8)]


def Encoder(input_img):
    Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
    Econv1_1 = BatchNormalization()(Econv1_1)
    Econv1_2 = Conv2D(16, (5, 5), activation='relu', padding='same',  name = "block1_conv2")(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)

    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)
    Econv2_2 = Conv2D(64, (5, 5), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)
    pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

    Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)
    Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

    encoded = Model(input = input_img, output = [pool3, Econv1_2, Econv2_2, Econv3_2] )

    encoded.summary()
    return encoded



def neck(input_layer):
    Nconv = Conv2D(256, (3,3),padding = "same", name = "neck1" )(input_layer)
    Nconv = BatchNormalization()(Nconv)
    Nconv = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv)
    Nconv = BatchNormalization()(Nconv)

    neck_model = Model(input_layer, Nconv)

    neck_model.summary()
    return neck_model



def Decoder(inp ):

    up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(inp[0])
    up1 = BatchNormalization()(up1)
    up1 = keras.layers.concatenate([up1, inp[3]],  axis=3, name = "merge_1")
    Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
    Upconv1_1 = BatchNormalization()(Upconv1_1)
    Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
    Upconv1_2 = BatchNormalization()(Upconv1_2)

    up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
    up2 = BatchNormalization()(up2)
    up2 = keras.layers.concatenate([up2, inp[2]], axis=3, name = "merge_2")
    Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
    Upconv2_1 = BatchNormalization()(Upconv2_1)
    Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
    Upconv2_2 = BatchNormalization()(Upconv2_2)

    up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
    up3 = BatchNormalization()(up3)
    up3 = keras.layers.concatenate([up3, inp[1]], axis=3, name = "merge_3")
    Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
    Upconv3_1 = BatchNormalization()(Upconv3_1)
    Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
    Upconv3_2 = BatchNormalization()(Upconv3_2)
        
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
    convnet = Model(input = inp, output =  decoded)
    convnet.summary()
    return convnet


############## INITIALIZE ############
#%%
x_shape = 360
y_shape = 480
channels = 7
input_img = Input(shape = (x_shape, y_shape, channels))

# encoder
encoded = Encoder(input_img)

# decoder
HG_ = Input(shape = (x_shape//(2**3),y_shape//(2**3),128))
conv1_l = Input(shape = (x_shape,y_shape,16))
conv2_l = Input(shape = (x_shape//(2**1),y_shape//(2**1),64))
conv3_l = Input(shape = (x_shape//(2**2),y_shape//(2**2),128))
decoded = Decoder( [HG_, conv1_l, conv2_l, conv3_l])

# Bottleneck
Neck_input = Input(shape = (x_shape//(2**3), y_shape//(2**3),128))
nck = neck(Neck_input)

# Combined
output_img = decoded([nck(encoded(input_img)[0]), encoded(input_img)[1], encoded(input_img)[2], encoded(input_img)[3]])
model= Model(input = input_img, output = output_img )
model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

def generator(frame_path, batchsize):
    if frame_path[-1] != '\\' and frame_path[-1] != '/':
        frame_path += '/'
    filelist = glob.glob(frame_path+'*.jpg')
    filetuples = [(filelist[i-2], filelist[i-1], filelist[i]) for i in range(2,len(filelist))]
    shuffle(filetuples)
    while True:
        for i in range(0,len(filetuples),batchsize):
            ft = filetuples[i:i+batchsize]
            inplist, outlist = list(), list()
            for f2, f1, f0 in ft:
                im2, im1, im0 = cv2.imread(f2), cv2.imread(f1), cv2.imread(f0)
                im0g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                inputtensor = np.stack((im2[:,:,0],im2[:,:,1],im2[:,:,2],im1[:,:,0],im1[:,:,1],im1[:,:,2],im0g), axis=-1)
                inplist.append(inputtensor)
                outlist.append(im0)
            inplist, outlist = np.array(inplist)/255.0, np.array(outlist)/255.0
            yield inplist, outlist
            

batchsize = 8

print('\n#############\n')

for epoch in range(3):
    shuffle(inpdirs)
    print('Main Epoch : %d'%(epoch))
    for frame_path in inpdirs:
        print(frame_path)
        history = model.fit_generator(generator(frame_path, batchsize), steps_per_epoch=int(9000/batchsize), epochs=1,
            verbose=1,
            validation_data=generator(frame_path, 3),
            validation_steps=1)
        model.save('model_tomNjerry.h5')
