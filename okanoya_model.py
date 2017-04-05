import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPool2D, Dense


# input dimensions: timesteps by frequencies
n_timesteps = 301
n_freqs = 112
n_train = 100
x_train = np.random.randn(n_train, n_timesteps, n_freqs, 1)
print(x_train.shape)

# output dimentions: the number of syllables + 1 for silence
n_syllables = 5
labels = np.random.randint(n_syllables, size=(n_train, 1))
y_train = keras.utils.to_categorical(labels, num_classes=n_syllables)
print(y_train.shape)
# the model
model = Sequential()
# first three layers
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                 input_shape=(n_timesteps, n_freqs, 1)))
model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))  # CCCP
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# fourth layer
model.add(Conv2D(240, kernel_size=(9, 11), activation='relu'))
# output layer
#model.add(Conv2D(n_syllables, kernel_size=(1, 1), activation='softmax'))
model.add(GlobalMaxPool2D())
model.add(Dense(n_syllables, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=10)
