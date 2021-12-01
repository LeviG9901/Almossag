import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import random, shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='rgb',
                                   class_mode=class_mode, target_size=target_size)


BS = 32

TS = (24, 24)
train_batch = generator('./input/training', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('./input/validation', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print("Number of images in a single batch of Training dataset =", SPE)
print("Number of images in a single batch of Validation dataset =", VS)


img,labels= next(train_batch)
print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),   # 1st layer
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),                                        # 2nd layer
    MaxPooling2D(pool_size=(1, 1)),
    # 32 convolution filters used each of size 3x3
    # again
    Conv2D(64, (3, 3), activation='relu'),                                        # 3rd layer
    MaxPooling2D(pool_size=(1, 1)),

    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling

    # randomly turn neurons on and off to improve convergence
    Dropout(0.25),
    # flatten since too many dimensions, we only want a classification output
    Flatten(),
    # fully connected to get all relevant data
    Dense(128, activation='relu'),
    # one more dropout for convergence' sake :)
    Dropout(0.5),
    # output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(train_batch, validation_data=valid_batch, epochs=50, steps_per_epoch=SPE, validation_steps=VS)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

model.save('./keras/eye_model.h5', overwrite=True)