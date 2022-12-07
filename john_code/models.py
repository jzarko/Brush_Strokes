import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt


new_kkanji_final_dataset_train = tf.keras.utils.image_dataset_from_directory(
    'directory here',
    validation_split=0.3,
    subset="training",
    image_size=(64, 64),
    batch_size=32,
    seed=132)

new_kkanji_final_dataset_val = tf.keras.utils.image_dataset_from_directory(
    'directory here',
    validation_split=0.3,
    subset="validation",
    image_size=(64, 64),
    batch_size=32,
    seed=132)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = new_kkanji_final_dataset_train.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = new_kkanji_final_dataset_val.cache().prefetch(buffer_size=AUTOTUNE)

#--------------------------------- Before Optuna ~83% val accuracy
john_model = models.Sequential()
john_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
john_model.add(layers.AveragePooling2D((2, 2)))
john_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
john_model.add(layers.AveragePooling2D((2, 2)))
john_model.add(layers.Flatten())
john_model.add(layers.Dense(64, activation='relu'))
john_model.add(layers.Dense(150))

john_model.summary()
john_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
john_history = john_model.fit(train_ds, 
                              epochs=20, 
                              callbacks=callback, 
                              validation_data=val_ds)
john_model.save('john_model')


# --------------------------------- Using Optuna results ~91% val accuracy
john_model = models.Sequential()
john_model.add(layers.Conv2D(59, (3, 3), activation='tanh', input_shape=(64, 64, 3)))
john_model.add(layers.AveragePooling2D((3, 3)))
john_model.add(layers.Conv2D(30, (4, 4), activation='tanh'))
john_model.add(layers.AveragePooling2D((4, 4)))
john_model.add(layers.Flatten())
john_model.add(layers.Dense(100, activation='tanh'))
john_model.add(layers.Dense(150))

john_model.summary()
john_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
john_history = john_model.fit(train_ds, 
                              epochs=20, 
                              callbacks=callback, 
                              validation_data=val_ds)
john_model.save('john_optuna_model')