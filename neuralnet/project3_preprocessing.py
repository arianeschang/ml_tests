#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:21:24 2017

@author: Olivier
"""
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Reshape, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D
from keras.optimizers import Adam

# dim_ordering='th' is to have the input format as follow : "3, 64, 64"

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        dim_ordering='th',
        fill_mode='nearest')

print('Importing data')
trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy') 
testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)


X0 = trainX[0:8000] 
X1 = trainX[8000:12000] 
X2 = trainX[12000:14080] 
X3 = trainX[14080:15120] 
X4 = trainX[15120:15921] 
X5 = trainX[15921:16630] 
X6 = trainX[16630:17270] 
X7 = trainX[17270:17851]
X8 = trainX[17851:18383] 
X9 = trainX[18383:18875] 
X10 = trainX[18875:19331]
X11 = trainX[19331:19757]
X12 = trainX[19757:20157]
X13 = trainX[20157:20533]
X14 = trainX[20533:20888]
X15 = trainX[20888:21224]
X16 = trainX[21224:21545]
X17 = trainX[21544:21848]
X18 = trainX[21848:22138]
X19 = trainX[22138:22415]
X20 = trainX[22415:22681]
X21 = trainX[22681:22937]
X22 = trainX[22937:23182]
X23 = trainX[23182:23418]
X24 = trainX[23418:23646]
X25 = trainX[23646:23866]
X26 = trainX[23866:24078]
X27 = trainX[24078:24284]
X28 = trainX[24284:24484]
X29 = trainX[24484:24677]
X30 = trainX[24677:24865]
X31 = trainX[24865:25047]
X32 = trainX[25047:25224]
X33 = trainX[25224:25396]
X34 = trainX[25396:25564]
X35 = trainX[25564:25728]
X36 = trainX[25728:25888]
X37 = trainX[25888:26044]
X38 = trainX[26044:26196]
X39 = trainX[26196:26344] 

y0 = [np.uint8(0) for i in range(len(X0))]
y1 = [np.uint8(1) for i in range(len(X1))]
y2 = [np.uint8(2) for i in range(len(X2))]
y3 = [np.uint8(3) for i in range(len(X3))]
y4 = [np.uint8(4) for i in range(len(X4))]
y5 = [np.uint8(5) for i in range(len(X5))]
y6 = [np.uint8(6) for i in range(len(X6))]
y7 = [np.uint8(7) for i in range(len(X7))]
y8 = [np.uint8(8) for i in range(len(X8))]
y9 = [np.uint8(9) for i in range(len(X9))]
y10 = [np.uint8(10) for i in range(len(X10))]
y11 = [np.uint8(11) for i in range(len(X11))]
y12 = [np.uint8(12) for i in range(len(X12))]
y13 = [np.uint8(13) for i in range(len(X13))]
y14 = [np.uint8(14) for i in range(len(X14))]
y15 = [np.uint8(15) for i in range(len(X15))]
y16 = [np.uint8(16) for i in range(len(X16))]
y17 = [np.uint8(17) for i in range(len(X17))]
y18 = [np.uint8(18) for i in range(len(X18))]
y19 = [np.uint8(19) for i in range(len(X19))]
y20 = [np.uint8(20) for i in range(len(X20))]
y21 = [np.uint8(21) for i in range(len(X21))]
y22 = [np.uint8(22) for i in range(len(X22))]
y23 = [np.uint8(23) for i in range(len(X23))]
y24 = [np.uint8(24) for i in range(len(X24))]
y25 = [np.uint8(25) for i in range(len(X25))]
y26 = [np.uint8(26) for i in range(len(X26))]
y27 = [np.uint8(27) for i in range(len(X27))]
y28 = [np.uint8(28) for i in range(len(X28))]
y29 = [np.uint8(29) for i in range(len(X29))]
y30 = [np.uint8(30) for i in range(len(X30))]
y31 = [np.uint8(31) for i in range(len(X31))]
y32 = [np.uint8(32) for i in range(len(X32))]
y33 = [np.uint8(33) for i in range(len(X33))]
y34 = [np.uint8(34) for i in range(len(X34))]
y35 = [np.uint8(35) for i in range(len(X35))]
y36 = [np.uint8(36) for i in range(len(X36))]
y37 = [np.uint8(37) for i in range(len(X37))]
y38 = [np.uint8(38) for i in range(len(X38))]
y39 = [np.uint8(39) for i in range(len(X39))]

trainX, X_val, trainY, y_val = train_test_split(trainX, trainY, test_size=0.4, random_state=0)

print('Generating new images')

trainX2 = list(trainX)
trainY2 = list(trainY)

i = 0
for X, Y in datagen.flow(X39,y39,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X38,y38,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X37,y37,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X36,y36,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X35,y35,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X34,y34,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X33,y33,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X32,y32,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X31,y31,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X30,y30,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break
i = 0
for X, Y in datagen.flow(X29,y29,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X28,y28,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X27,y27,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X26,y26,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X25,y25,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X24,y24,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X23,y23,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X22,y22,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X21,y21,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X20,y20,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1800:
        break

i = 0
for X, Y in datagen.flow(X19,y19,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1700:
        break

i = 0
for X, Y in datagen.flow(X18,y18,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1700:
        break

i = 0
for X, Y in datagen.flow(X17,y17,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1700:
        break

i = 0
for X, Y in datagen.flow(X16,y16,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1700:
        break

i = 0
for X, Y in datagen.flow(X15,y15,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1700:
        break

i = 0
for X, Y in datagen.flow(X14,y14,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1600:
        break

i = 0
for X, Y in datagen.flow(X13,y13,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1600:
        break

i = 0
for X, Y in datagen.flow(X12,y12,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1600:
        break

i = 0
for X, Y in datagen.flow(X11,y11,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1600:
        break

i = 0
for X, Y in datagen.flow(X10,y10,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1500:
        break

i = 0
for X, Y in datagen.flow(X9,y9,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1500:
        break

i = 0
for X, Y in datagen.flow(X8,y8,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1500:
        break

i = 0
for X, Y in datagen.flow(X7,y7,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1400:
        break

i = 0
for X, Y in datagen.flow(X6,y6,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1400:
        break

i = 0
for X, Y in datagen.flow(X5,y5,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1300:
        break

i = 0
for X, Y in datagen.flow(X4,y4,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1200:
        break

i = 0
for X, Y in datagen.flow(X3,y3,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1000:
        break

i = 0
for X, Y in datagen.flow(X2,y2,batch_size=1):
    trainX2.append(X[0])
    trainY2.append(Y[0])
    i+=1
    if i > 1000:
        break
trainX2 = np.float32(trainX2)
trainY2 = np.float32(trainY2)


## to visualize only
#from matplotlib import pyplot as plt
#plt.imshow(trainX[1].transpose(2,1,0))
#plt.show()

#To save the generated images dataset

#np.save('trainX2_datagen', trainX2)
#np.save('trainY2_datagen', trainY2)
#np.save('X_val_datagen', X_val)
#np.save('y_val_datagen', y_val)

# to load the generated images dataset

#trainX2 = np.load('trainX2_datagen.npy')
#trainY2 = np.load('trainY2_datagen.npy')
#X_val = np.load('X_val_datagen.npy')
#y_val = np.load('y_val_datagen.npy')

print('Spliting data into 2 sets (training & validation)')

X_train, X_val2, y_train, y_val2 = train_test_split(trainX2, trainY2, test_size=0.4, random_state=1)

y_train3 = pd.DataFrame(y_train)
y_train2 = pd.DataFrame(y_train)

print('Converting classes to binary (dummy variables)')

y_train2[0]=np.where(y_train3==0,1,0)
y_train2[1]=np.where(y_train3==1,1,0)
y_train2[2]=np.where(y_train3==2,1,0)
y_train2[3]=np.where(y_train3==3,1,0)
y_train2[4]=np.where(y_train3==4,1,0)
y_train2[5]=np.where(y_train3==5,1,0)
y_train2[6]=np.where(y_train3==6,1,0)
y_train2[7]=np.where(y_train3==7,1,0)
y_train2[8]=np.where(y_train3==8,1,0)
y_train2[9]=np.where(y_train3==9,1,0)
y_train2[10]=np.where(y_train3==10,1,0)
y_train2[11]=np.where(y_train3==11,1,0)
y_train2[12]=np.where(y_train3==12,1,0)
y_train2[13]=np.where(y_train3==13,1,0)
y_train2[14]=np.where(y_train3==14,1,0)
y_train2[15]=np.where(y_train3==15,1,0)
y_train2[16]=np.where(y_train3==16,1,0)
y_train2[17]=np.where(y_train3==17,1,0)
y_train2[18]=np.where(y_train3==18,1,0)
y_train2[19]=np.where(y_train3==19,1,0)
y_train2[20]=np.where(y_train3==20,1,0)
y_train2[21]=np.where(y_train3==21,1,0)
y_train2[22]=np.where(y_train3==22,1,0)
y_train2[23]=np.where(y_train3==23,1,0)
y_train2[24]=np.where(y_train3==24,1,0)
y_train2[25]=np.where(y_train3==25,1,0)
y_train2[26]=np.where(y_train3==26,1,0)
y_train2[27]=np.where(y_train3==27,1,0)
y_train2[28]=np.where(y_train3==28,1,0)
y_train2[29]=np.where(y_train3==29,1,0)
y_train2[30]=np.where(y_train3==30,1,0)
y_train2[31]=np.where(y_train3==31,1,0)
y_train2[32]=np.where(y_train3==32,1,0)
y_train2[33]=np.where(y_train3==33,1,0)
y_train2[34]=np.where(y_train3==34,1,0)
y_train2[35]=np.where(y_train3==35,1,0)
y_train2[36]=np.where(y_train3==36,1,0)
y_train2[37]=np.where(y_train3==37,1,0)
y_train2[38]=np.where(y_train3==38,1,0)
y_train2[39]=np.where(y_train3==39,1,0)
y_train2 = np.array(y_train2)

y_val3 = pd.DataFrame(y_val)
y_val2 = pd.DataFrame(y_val)

y_val2[0]=np.where(y_val3==0,1,0)
y_val2[1]=np.where(y_val3==1,1,0)
y_val2[2]=np.where(y_val3==2,1,0)
y_val2[3]=np.where(y_val3==3,1,0)
y_val2[4]=np.where(y_val3==4,1,0)
y_val2[5]=np.where(y_val3==5,1,0)
y_val2[6]=np.where(y_val3==6,1,0)
y_val2[7]=np.where(y_val3==7,1,0)
y_val2[8]=np.where(y_val3==8,1,0)
y_val2[9]=np.where(y_val3==9,1,0)
y_val2[10]=np.where(y_val3==10,1,0)
y_val2[11]=np.where(y_val3==11,1,0)
y_val2[12]=np.where(y_val3==12,1,0)
y_val2[13]=np.where(y_val3==13,1,0)
y_val2[14]=np.where(y_val3==14,1,0)
y_val2[15]=np.where(y_val3==15,1,0)
y_val2[16]=np.where(y_val3==16,1,0)
y_val2[17]=np.where(y_val3==17,1,0)
y_val2[18]=np.where(y_val3==18,1,0)
y_val2[19]=np.where(y_val3==19,1,0)
y_val2[20]=np.where(y_val3==20,1,0)
y_val2[21]=np.where(y_val3==21,1,0)
y_val2[22]=np.where(y_val3==22,1,0)
y_val2[23]=np.where(y_val3==23,1,0)
y_val2[24]=np.where(y_val3==24,1,0)
y_val2[25]=np.where(y_val3==25,1,0)
y_val2[26]=np.where(y_val3==26,1,0)
y_val2[27]=np.where(y_val3==27,1,0)
y_val2[28]=np.where(y_val3==28,1,0)
y_val2[29]=np.where(y_val3==29,1,0)
y_val2[30]=np.where(y_val3==30,1,0)
y_val2[31]=np.where(y_val3==31,1,0)
y_val2[32]=np.where(y_val3==32,1,0)
y_val2[33]=np.where(y_val3==33,1,0)
y_val2[34]=np.where(y_val3==34,1,0)
y_val2[35]=np.where(y_val3==35,1,0)
y_val2[36]=np.where(y_val3==36,1,0)
y_val2[37]=np.where(y_val3==37,1,0)
y_val2[38]=np.where(y_val3==38,1,0)
y_val2[39]=np.where(y_val3==39,1,0)
y_val2 = np.float64(y_val2)


#******************************************************************************************
#Complex model (1 hidden layer)
model = Sequential()

# Model is 64x64 pixel (output dim) and each observation has 12288 "variable" (input_dim)
model.add(Dense(output_dim=64, input_shape=(3, 64, 64)))

# hidden layers
model.add(Conv2D(16, 3, 3, activation='relu', dim_ordering = 'th', subsample=(2,2)))
model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = 'th'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(output_dim=128, activation='relu'))
model.add(Dropout(0.5)) # Dropout is to eliminate overfitting

model.add(Dense(output_dim=40, activation='softmax')) # 40 classes (softmax)
   
#Compile 
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print('Fitting model')
model.fit(X_train, y_train, nb_epoch=25, batch_size=100)

# Save model
model.save('model_2')

# Evaluate validation set (10% data, 50 epochs - acc : 31.1%, overfit with 50 epochs)
print('Evaluate the validation set')
model.evaluate(X_val,y_val)
#******************************************************************************************

# Prediction
print('Predicting')
pred = model.predict(testX)
pred=pd.DataFrame(pred)
pred1=pred.idxmax(axis=1)
pred1=pd.DataFrame(pred1)
pred1.columns = ['class']
pred1.to_csv("project3_keras.csv", index = True, index_label = 'id')

