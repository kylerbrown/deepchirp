import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
def get_model(model_name, n_timesteps, n_freqs, n_cats):
    if model_name == 'okanoya':
        return okanoya_model(n_timesteps, n_freqs, n_cats)
    if model_name == 'dummy':
        return dummy_model(n_timesteps, n_freqs, n_cats)
    else:
        raise KeyError("could not find model {}".format(model_name))

def dummy_model(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(n_cats, activation='softmax'))
    return model

def okanoya_model(n_freqs, n_timesteps, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    model = Sequential()
    # first three layers
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_freqs, n_timesteps, 1)))
#    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))  # CCCP
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
#    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#    model.add(Conv2D(16, kernel_size=(4, 3), activation='relu'))
#    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # fourth layer
    model.add(Conv2D(100, kernel_size=(4, 4), activation='relu'))
    # output layer
    #model.add(Conv2D(n_syllables, kernel_size=(1, 1), activation='softmax'))
    model.add(Flatten())
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01,
                                                 decay=1e-6,
                                                 momentum=0.9,
                                                 nesterov=True),
                  metrics=['accuracy'])
    return model
