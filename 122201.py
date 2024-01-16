import tensorflow
import keras.utils as image_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import PredefinedSplit, train_test_split
import PIL.ImageOps
import pandas as pd

#loading dataset
df=pd.read_csv("dataset/kaggle/digit-recognizer/train.csv")
x=df.drop('label',axis=1)/255#/255 to normalize
y=df['label']
tx,sx,ty,sy=train_test_split(x,y,test_size=0.3,random_state=65)
#samplex=x.iloc[0][0:784].values.reshape(28,28) this script aim to read the image from human eye
num_categories=10
sy=image_utils.to_categorical(sy, num_categories)
ty=image_utils.to_categorical(ty, num_categories)
#start training
model=Sequential()
model.add(Dense(units=512,activation="relu",input_shape=(784,)))
model.add(Dense(units=256,activation="relu",input_shape=(512,)))
model.add(Dropout(0.25))
model.add(Dense(units=256,activation="sigmoid"))
model.add(Dense(units=128,activation="relu",input_shape=(256,)))
model.add(Dropout(0.25))
model.add(Dense(units=64,activation="tanh",input_shape=(128,)))
model.add(Dense(units=32,activation="relu",input_shape=(64,)))
model.add(Dense(units=10,activation="softmax"))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(tx,ty,epochs=10,verbose=1,validation_data=(sx, sy))
#test the data:)
def testdata(num):
    print(np.argmax(model.predict(np.array([x.iloc[num]]))))
    plt.imshow(x.iloc[num][0:784].values.reshape(28,28) )

#lgtm
testsubmit=pd.read_csv("dataset/kaggle/digit-recognizer/test.csv")
pred=model.predict(testsubmit)
sumbitdf=pd.DataFrame(columns=['ImageId','Label'])
sumbitdf['ImageId']=range(1,28001)
sumbitdf['Label']=np.argmax(pred, axis=1)
sumbitdf.to_csv('for_submission_2023122301.csv', index=False)