import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as sci_wav
import warnings
import librosa as lb
warnings.filterwarnings('ignore')
from keras.models import Sequential,load_model
from keras.layers import MaxPooling2D,Conv2D,Flatten,Activation,Dense,Dropout,BatchNormalization
from sklearn import preprocessing
from keras.utils import to_categorical
from IPython.display import clear_output

root="C:\\Users\\sarth\\Desktop\\Woof and Meow\\cats_dogs\\"

#Labelling of data(y).
y=[1 if 'cat' in i else 0 for i in os.listdir(root)]

#def number_of_files(test_size):
x_train,x_test,y_train,y_test=train_test_split(os.listdir(root),y,test_size=0.25)
#print("Test Split: {}".format(test_size))
print('Dataset has {} cat files and {} dog files.'.format(sum(y),len(y)-sum(y)))
print('X_train has {} cat files and {} dog files.'.format(sum(y_train),len(y_train)-sum(y_train)))
print('X_test has {} cat files and {} dog files.'.format(sum(y_test),len(y_test)-sum(y_test)))
    #return x_train

#Splitting of data into train and test files.
#def read_wav_files(wav_files):
    
    #if not isinstance(wav_files,list):
        #wav_files=[wav_files]
    #return [sci_wav.read(root+f)[1] for f in wav_files]

#print(read_wav_files(x_train))
#x_train,x_test=map(read_wav_files,[x_train,x_test])

#all_files=read_wav_files(os.listdir(root))
#all_files_cat=[a for a,b in zip(all_files,y) if b==1]
#all_files_dog=[a for a,b in zip(all_files,y) if b==0]
#all_files_cat=np.concatenate(all_files_cat)
#all_files_dog=np.concatenate(all_files_dog)

#Total duration of individual splits.
def len_of_files(sample_rate):
    cat_time=len(all_files_cat)/sample_rate
    dog_time=len(all_files_dog)/sample_rate
    #print('Overall, there are {:.2f}s of cat and {:.2f}s of dog sound'.format(cat_time,dog_time))

#print(len_of_files(22050))

#Audio Features    
def features_go(file):
    x,sample_rate=lb.load(file,duration=5)
    stft = np.abs(lb.stft(x))
    mfccs = np.mean(lb.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(lb.feature.melspectrogram(x, sr=sample_rate).T,axis=0)
    contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(lb.feature.tonnetz(y=lb.effects.harmonic(x),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


X_train=[]
i=0
while i<len(x_train):
    print("File number: ",i)
    filename = root+str(x_train[i])
    mfccs,chroma,mel,contrast,tonnetz=features_go(filename)
    features = []
    features.append(np.mean(mfccs))
    features.append(np.mean(chroma))
    features.append(np.mean(mel))
    features.append(np.mean(contrast))
    features.append(np.mean(tonnetz))
    X_train.append(features)
    i+=1
    

X_train=np.asarray(X_train)
Y_train=np.asarray(y_train)

model=Sequential()
model.add(Dense(500,input_shape=(5,)))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(optimizer = 'adam', metrics=['accuracy'], loss = 'binary_crossentropy')

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
        
plot_losses = PlotLosses()

model.fit(X_train,Y_train,epochs=500,callbacks=[plot_losses],verbose=1)

X_test=[]
i=0
while i<len(x_test):
    print("File number: ",i)
    filename = root+str(x_test[i])
    mfccs,chroma,mel,contrast,tonnetz=features_go(filename)
    features = []
    features.append(np.mean(mfccs))
    features.append(np.mean(chroma))
    features.append(np.mean(mel))
    features.append(np.mean(contrast))
    features.append(np.mean(tonnetz))
    X_test.append(features)
    i+=1
    
y_pred=model.predict(np.asarray(X_test))

i=0
while i<len(y_pred):
    if y_pred[i]<=0.5:
        y_pred[i]=0
    else: 
        y_pred[i]=1
    i+=1
y_test=np.asarray(y_pred)
y_pred=y_pred.reshape([-1])
    
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
       
        
        #Normalization can be applied by setting `normalize=True`.
        
        if normalize:
           cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
           print("Normalized confusion matrix")
        else:
           print('Confusion matrix, without normalization')
        
        print(cm)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
           plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['cat','dog'],title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['cat','dog'], normalize=True,title='Normalized confusion matrix')

plt.show()
