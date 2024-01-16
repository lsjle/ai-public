from keras.applications import VGG16
import keras
import keras.utils as image_utils
model=VGG16(weights="imagenet")
#https://mobiledev.tw/animal-classification/
#should be done but not yet
base_model=keras.applications.VGG16(weights='imagenet',input_shape=(224,224,3),include_top=False)
base_model.summary()
base_model.trainable=False
inputs=keras.Input(shape=(224,224,3))
x=base_model(inputs,training=False)
x_afterPooling=keras.layers.GlobalAveragePooling2D()(x)
outputs=keras.layers.Dense(1)(x_afterPooling)
custom_model=keras.Model(inputs,outputs)
custom_model.summary()
custom_model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=[keras.metrics.BinaryAccuracy()])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(samplewise_center=True,rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=False)
#auto generate data with exist image that is provided
train_data=datagen.flow_from_directory('dataset/IsBoOrNot/IsBoOrNot/train',target_size=(224,224),color_mode='rgb',class_mode='binary',batch_size=8)
valid_data=datagen.flow_from_directory('dataset/IsBoOrNot/IsBoOrNot/valid',target_size=(224,224),color_mode='rgb',class_mode='binary',batch_size=8)
custom_model.fit(train_data,steps_per_epoch=12,validation_data=valid_data,validation_steps=4,epochs=20)