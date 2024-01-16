#copy from https://gist.github.com/ryanchung403/2a51141ef3e0d4a52bada9cf9b051f75
# Import dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
print(tf.__file__)
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as image_utils
import PIL.ImageOps

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

plt.imshow(X_train[0], cmap="gray")
y_train[0]

X_train_1D = X_train.reshape(60000, 784)
X_valid_1D = X_valid.reshape(10000, 784)

X_train_1D_normal = X_train_1D / 255#check color to only! b&w
X_valid_1D_normal = X_valid_1D / 255

num_categories = 10#0-10
y_train
y_train_category = image_utils.to_categorical(y_train, num_categories)
y_valid_category = image_utils.to_categorical(y_valid, num_categories)
# 測試看看 y_train_category[0]

model = Sequential()
model.add(Dense(units=512, activation="relu", input_shape=(784,)))#this is a input layer
model.add(Dense(units=512, activation="relu"))#this is a hidden layer
model.add(Dense(units=256, activation="sigmoid",input_shape=(512,)))#this is a hidden layer
model.add(Dense(units=256, activation="tanh"))#this is a hidden layer
model.add(Dense(units=128, activation="relu",input_shape=(256,)))#this is a hidden layer
model.add(Dense(units=10, activation="softmax"))#this will be the exit (output layer)
model.summary()

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train_1D_normal,
    y_train_category,
    epochs=10,
    verbose=1,
    validation_data=(X_valid_1D_normal, y_valid_category),
)

# model.predict(X_valid_1D_normal[0])
# model.predict([X_valid_1D_normal[0]])
# model.predict(np.array(X_valid_1D_normal[0]))
#above in correct test form

model.predict(np.array([X_valid_1D_normal[0]]))

np.argmax(model.predict(np.array([X_valid_1D_normal[0]])))

plt.imshow(X_valid[0])


def predict_number(image_path):
    if "http" in image_path:
        image_path = image_utils.get_file(origin=image_path)
    test_image = image_utils.load_img(
        image_path, color_mode="grayscale", target_size=(28, 28)
    )
    test_image_inverted = PIL.ImageOps.invert(test_image)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(test_image, cmap="gray")
    axarr[1].imshow(test_image_inverted, cmap="gray")

    test_image_array = image_utils.img_to_array(test_image_inverted)
    test_image_array_1D = test_image_array.reshape(1, 784)
    test_image_array_1D_normal = test_image_array_1D / 255
    return np.argmax(model.predict(test_image_array_1D_normal))


predict_number("4.jpg")
predict_number(
    "https://www.e-tennis.com/pub/media/catalog/product/cache/f5f3da80ad7b670245aea7e970662954/n/u/number-4.jpg"
)

np.argmax(model.predict(np.array([X_valid_1D_normal[3]])))
plt.imshow(X_valid[3], cmap="gray")