#%%
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


#%%
inpdirs = ['./ColorFrames%d'%(i) for i in range(1,9)]


#%%
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


#%%
def neck(input_layer):
    Nconv = Conv2D(256, (3,3),padding = "same", name = "neck1" )(input_layer)
    Nconv = BatchNormalization()(Nconv)
    Nconv = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv)
    Nconv = BatchNormalization()(Nconv)

    neck_model = Model(input_layer, Nconv)

    neck_model.summary()
    return neck_model


#%%
def Decoder(inp):
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
        
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same', name = "Ouput_layer")(Upconv3_2)
    convnet = Model(input = inp, output =  [decoded, Upconv1_2, Upconv2_2])
    convnet.summary()
    return convnet



#%%
def DualConnect(layers_11, layers_12, layers_21, layers_22, outlayers1, outlayers2):
    concat1 = keras.layers.concatenate([layers_11, layers_12], axis=3, name="dualmerge1")
    conv1 = Conv2D(128, (3,3), activation='relu', padding='same', name='dual_conv1')(concat1)
    conv1 = BatchNormalization()(conv1)

    up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "dual_upconv2")(conv1)
    up2 = BatchNormalization()(up2)
    concat2 = keras.layers.concatenate([layers_21, up2, layers_22], axis=3, name="dualmerge2")
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', name='dual_conv2')(concat2)
    conv2 = BatchNormalization()(conv2)

    up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "dual_upconv3")(conv2)
    up3 = BatchNormalization()(up3)
    conv3 = Conv2D(3,(3,3), activation='relu', padding='same', name='dual_conv3')(up3)
    conv3 = BatchNormalization()(conv3)
    concat3 = keras.layers.concatenate([outlayers1, conv3, outlayers2], axis=3, name="dualmerge3")

    conv4 = Conv2D(3, (3,3), activation='sigmoid', padding='same', name='dual_conv4')(concat3)

    connectnet = Model(input = [layers_11, layers_12, layers_21, layers_22, outlayers1, outlayers2],
        output=conv4)
    connectnet.summary()
    return connectnet



############## INITIALIZE ############
#%%
x_shape = 360
y_shape = 480
channels = 7
input_img7 = Input(shape = (x_shape, y_shape, channels))
input_img1 = Input(shape = (x_shape, y_shape, 1))
# input_img1 = keras.layers.convolutional.Cropping3D(cropping=((1,1),(1,1),(1,1)))(input_img7)


#%%
# ------------------7 channel--------------------
# encoder
encoded7 = Encoder(input_img7)
# decoder
HG_7 = Input(shape = (x_shape//(2**3),y_shape//(2**3),128))
conv1_l7 = Input(shape = (x_shape,y_shape,16))
conv2_l7 = Input(shape = (x_shape//(2**1),y_shape//(2**1),64))
conv3_l7 = Input(shape = (x_shape//(2**2),y_shape//(2**2),128))
decoded7 = Decoder( [HG_7, conv1_l7, conv2_l7, conv3_l7])
# Bottleneck
Neck_input7 = Input(shape = (x_shape//(2**3), y_shape//(2**3),128))
nck7 = neck(Neck_input7)
# out
out7 = decoded7([nck7(encoded7(input_img7)[0]), encoded7(input_img7)[1], encoded7(input_img7)[2], encoded7(input_img7)[3]])


#%%
# -------------------1 channel--------------------
#encoder
encoded1 = Encoder(input_img1)
# decoder
HG_1 = Input(shape = (x_shape//(2**3),y_shape//(2**3),128))
conv1_l1 = Input(shape = (x_shape,y_shape,16))
conv2_l1 = Input(shape = (x_shape//(2**1),y_shape//(2**1),64))
conv3_l1 = Input(shape = (x_shape//(2**2),y_shape//(2**2),128))
decoded1 = Decoder( [HG_1, conv1_l1, conv2_l1, conv3_l1])
# Bottleneck
Neck_input1 = Input(shape = (x_shape//(2**3), y_shape//(2**3),128))
nck1 = neck(Neck_input1)
# out
out1 = decoded1([nck1(encoded1(input_img1)[0]), encoded1(input_img1)[1], encoded1(input_img1)[2], encoded1(input_img1)[3]])



#%%
# -----------------------Combining 1 & 7 channel networks-------------------------
outlayers_inp0 = Input(shape = (x_shape, y_shape, 3))
outlayers_inp1 = Input(shape = (x_shape, y_shape, 3))
layers_1_inp0 = Input(shape = (x_shape//2, y_shape//2, 64))
layers_1_inp1 = Input(shape = (x_shape//2, y_shape//2, 64))
layers_2_inp0 = Input(shape = (x_shape//4, y_shape//4, 128))
layers_2_inp1 = Input(shape = (x_shape//4, y_shape//4, 128))
DC = DualConnect(layers_2_inp0, layers_2_inp1,
    layers_1_inp0, layers_1_inp1,
    outlayers_inp0, outlayers_inp1)

fout = DC([out7[1], out1[1],
    out7[2], out1[2],
    out7[0], out1[0]])

#%%
model= Model(input = [input_img7, input_img1], output = fout )
model.summary()
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


#%%
def generator(frame_path, batchsize):
    if frame_path[-1] != '\\' and frame_path[-1] != '/':
        frame_path += '/'
    filelist = glob.glob(frame_path+'*.jpg')
    # prepare triplets
    filetuples = [[filelist[i-2], filelist[i-1], filelist[i]] for i in range(2,len(filelist))]
    shuffle(filetuples)
    l = len(filetuples)
    filetuples = filetuples[:int(l/18)]+filetuples
    for i in range(int(l/18)):
        filetuples[i][0]= '$$'
    # second part
    shuffle(filetuples)
    l = len(filetuples)
    filetuples = filetuples[:int(l/15)]+filetuples
    for i in range(int(l/15)):
        filetuples[i][0], filetuples[i][1] = filetuples[-i][0], filetuples[int(-i-(l/15))][1]
    # last shuffle
    shuffle(filetuples)
    l = len(filetuples)
    # yielding part
    while True:
        for i in range(0,len(filetuples),batchsize):
            ft = filetuples[i:i+batchsize]
            inplist, outlist, singlelist = list(), list(), list()
            for f2, f1, f0 in ft:
                if f2 != '$$':
                    im2, im1, im0 = cv2.imread(f2), cv2.imread(f1), cv2.imread(f0)
                else:
                    im2, im1 = np.zeros((x_shape, y_shape, 3), dtype=np.uint8), np.zeros((x_shape, y_shape, 3), dtype=np.uint8)
                    im0 = cv2.imread(f0)
                im0g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                inputtensor = np.stack((im2[:,:,0],im2[:,:,1],im2[:,:,2],im1[:,:,0],im1[:,:,1],im1[:,:,2],im0g), axis=-1)
                inplist.append(inputtensor)
                singlelist.append(inputtensor[:,:,6:])
                outlist.append(im0)
            inplist, outlist = np.array(inplist)/255.0, np.array(outlist)/255.0
            singlelist = np.array(singlelist)/255.0
            yield {'input_1':inplist,'input_2':singlelist}, outlist
            

batchsize = 8

print('\n#############\n')

for epoch in range(2):
    shuffle(inpdirs)
    print('\n\nMain Epoch : %d\n\n'%(epoch))
    for frame_path in inpdirs:
        print(frame_path)
        history = model.fit_generator(generator(frame_path, batchsize), steps_per_epoch=int(10000/batchsize), epochs=1,
            verbose=1,
            validation_data=generator(frame_path, 3),
            validation_steps=1)
        model.save('model_tomNjerry_nfinal.h5')
