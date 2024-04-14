# Training Image Classification model

import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

fish_data= os.path.join('.', '5classes')

height,width=240,135

training_batch_size=32

train_set = tf.keras.preprocessing.image_dataset_from_directory(
  fish_data,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(height,width),
  batch_size=training_batch_size)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
  fish_data,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(height, width),
  batch_size=training_batch_size)

image_cat = train_set.class_names
print(image_cat)

########################################################################################################################################
#Training the Model
#######################################################################################################################################

resnet_model = Sequential()

imported_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(240,135,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
for layer in imported_model.layers:
        layer.trainable = False

resnet_model.add(imported_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dropout(0.3))
resnet_model.add(k.layers.BatchNormalization())
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dropout(0.3))
resnet_model.add(k.layers.BatchNormalization())
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dropout(0.3))
resnet_model.add(k.layers.BatchNormalization())
resnet_model.add(Dense(5, activation='softmax'))

resnet_model.compile(optimizer=Adam(learning_rate=0.00005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.5)
])

train_set = train_set.map(lambda x, y: (data_augmentation(x, training=True), y))

history = resnet_model.fit(
  train_set,
  validation_data=validation_set,
  epochs=30
)

resnet_model.save("__5class_custom_model_30.keras")

# Plot Accuracy training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss training history
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()






