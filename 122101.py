#english word recognition
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.utils as image_utils
from sklearn.model_selection import train_test_split
import PIL.ImageOps
#our datasize will be 128**
df=pd.read_csv("dataset/kaggle/train.csv")
xtrain,xtest,ytrain,ytest=train_test_split(df['filepaths'],df['Font'],test_size=0.3,random_state=83)
xtrainimg =[]
for filepath in xtrain:
    # Open the image file
    img = image_utils.load_img(filepath,color_mode="grayscale",target_size=(128,128))
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # Append the array to the list
    xtrainimg.append(img_array)
#load image aka image preprocessing
# test_image = image_utils.load_img(
#     image_path, color_mode="grayscale", target_size=(128, 128)
# )
xtestimg = []
for filepath in xtest:
    # Open the image file
    img = image_utils.load_img(filepath,color_mode="grayscale",target_size=(128,128))
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # Append the array to the list
    xtestimg.append(img_array)
npxtrainimg=np.array(xtrainimg)
npxtestimg=np.array(xtestimg)
X_train_1D = npxtrainimg.reshape(44094, 16384)
X_valid_1D = npxtestimg.reshape(18898, 16384)
X_train_1D_normal = X_train_1D / 255#check color to only! b&w
X_valid_1D_normal = X_valid_1D / 255
num_categories=62
def replace_letters_with_numbers(input_list):
    result_list = []

    for original_string in input_list:
        modified_string = ""
        for char in original_string:
            if 'a' <= char <= 'z':
                modified_string += str(ord(char) - ord('a') + 11)
            elif 'A' <= char <= 'Z':
                modified_string += str(ord(char) - ord('A') + 11)
            else:
                modified_string += char

        result_list.append(modified_string)

    return result_list
ytrain = replace_letters_with_numbers(ytrain)
ytest = replace_letters_with_numbers(ytest)
y_train_category = image_utils.to_categorical(ytrain, num_categories)
y_valid_category = image_utils.to_categorical(ytest, num_categories)#what?????????
#start doing model
model=Sequential()
model.add(Dense(units=4096,activation="relu",input_shape=(16384,)))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1024,activation="relu",input_shape=(4096,)))
model.add(Dense(units=256, activation="tanh",input_shape=(1024,)))
model.add(Dense(units=128, activation="relu",input_shape=(256,)))
model.add(Dense(units=62, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train_1D_normal,
    y_train_category,
    epochs=10,
    verbose=1,
    validation_data=(X_valid_1D_normal, y_valid_category),
)
model.predict(np.array([X_valid_1D_normal[0]]))

np.argmax(model.predict(np.array([X_valid_1D_normal[0]])))

plt.imshow(X_valid[0])