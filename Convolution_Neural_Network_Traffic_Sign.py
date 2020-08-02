import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from time import time
data=[]
labels=[]

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width*channels


# following accesses images from the directory Train and puts them into numpy arrays. 
# Validation images are taken from within the Train folder itself. The test accuracy and confusion matrix
# Will be done using the images in the Test folder. 
start = time() # start timer

for i in range(classes) :
    path = "train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")
            
Cells=np.array(data)
labels=np.array(labels)

#Randomize the order of the input images
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

X_train=Cells[(int)(0.2*len(labels)):]
X_val=Cells[:(int)(0.2*len(labels))]
X_train=X_train.astype('float32')/255
X_val=X_val.astype('float32')/255
y_train=labels[(int)(0.2*len(labels)):]
y_val=labels[:(int)(0.2*len(labels))]


#CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D,Dense,Flatten,Dropout,ConvLSTM2D 

model=Sequential();
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape=X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2,2))) 
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())

model.add(Dense(43,activation='softmax')) # Output Layer

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=32,epochs=10,verbose=1)

plt.figure(0)
plt.plot(history.history['accuracy'],label='training Accuracy')
plt.plot(history.history['val_accuracy'],label='val Accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# This deals with the images in the Test Folder. 
y_test=pd.read_csv('Test.csv')
labels = y_test['Path'].as_matrix()
y_test=y_test['ClassId'].values


data=[]

for f in labels:
    image=cv2.imread("{0}".format(f))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

X_test=np.array(data)
X_test = X_test.astype('float32')/255  
pred = model.predict_classes(X_test)
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

print("The accuracy score is: ", accuracy_score(y_test, pred))
print("The confusion matrix is: \n", confusion_matrix(y_test, pred)) # Gives the number of True_positive/False_positives/True_negatives/False_nagatives
print("The F1 Score is: ", f1_score(y_test, pred, average = 'macro'))

end = time()

print("Time taken: ", end - start)
