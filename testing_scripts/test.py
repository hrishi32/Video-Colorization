#!/usr/bin/env python
# coding: utf-8

import keras
from keras.models import *
import numpy as np
import cv2
import glob
import sys
import os


def frameExtractor(path):
    videoObject = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = videoObject.read()  
        yield image[:,:480,:]/255.0


print("Model loading")
model = load_model(sys.argv[1])
print("Model loaded")
it = frameExtractor('./TomAndJerry9.mp4')

gt = list()
for i in range(2000):
    im = next(it)
    gt.append(im)
gt = np.array(gt)

preds = np.copy(gt)
abc = preds

for i in range(2,len(gt)):
    if i%20==0: print(i)
    gr = gt[i-2:i+1]
    abc = gr[2]
    gray = cv2.cvtColor((abc*255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)/255.0
    inpu = np.stack((preds[0][:,:,0],preds[0][:,:,1],preds[0][:,:,2],preds[1][:,:,0],preds[1][:,:,1],preds[1][:,:,2],gray), axis=-1)
    # inpu_1 = gray.reshape((360, 480, 1))
    # input_1_list = np.array([inpu_1],dtype=float)
    input_list = np.array([inpu], dtype=float)
    preds[i] = model.predict(input_list)[0]

dirname = sys.argv[1]+'d/'
if not os.path.exists(dirname):
        os.makedirs(dirname)


for i in range(2000):
    temp = np.zeros(shape=(360,480*2,3))
    temp[:,:480,:], temp[:,480:,:] = preds[i], gt[i]
    # temp[:,:,0], temp[:,:,1], temp[:,:,2] = temp[:,:,2], temp[:,:,1], temp[:,:,0]
    cv2.imwrite(dirname+str(i)+'.png',(temp*255).astype(np.uint8))
