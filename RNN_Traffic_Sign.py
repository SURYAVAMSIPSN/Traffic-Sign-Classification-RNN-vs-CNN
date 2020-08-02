import numpy as np
import pandas as pd
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
n_inputs = height * width


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
            image=cv2.imread(path+a, 0)
            size_image = cv2.resize(image, (height, width))
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
X_val=X_val.astype('float32')/255 # Normalization
y_train=labels[(int)(0.2*len(labels)):]
y_val=labels[:(int)(0.2*len(labels))]
# # # Up until here it's the same, regardless of the type of network to be used. 

#Rec-NN
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM # long short term memory


model = Sequential()

model.add(LSTM(32, input_shape = X_train.shape[1:], activation = 'relu', return_sequences = True))
model.add(Dropout(0.25))

model.add(LSTM(64, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(43, activation = 'softmax'))

opt = keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

model.compile(loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

history = model.fit(X_train,
          y_train,
          epochs=10,
          validation_data=(X_val, y_val))

# # # From here on, it's the same regardless of the type of network
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
    image=cv2.imread("{0}".format(f), 0)
    size_image = cv2.resize(image, (height, width))
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
