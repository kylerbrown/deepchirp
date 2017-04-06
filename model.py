import keras
from keras.models import Sequential, Model
from keras.layers import Permute, Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout

def get_model(model_name, n_timesteps, n_freqs, n_cats, **kwargs):
    print(model_name)
    if model_name == 'recurrent':
        return recurrent(n_cats, **kwargs)
    if model_name == 'okanoya_r':
        return okanoya_r_model(n_timesteps, n_freqs, n_cats, **kwargs)
    if model_name == 'okanoya':
        return okanoya_model(n_timesteps, n_freqs, n_cats)
    if model_name == 'dumb_r':
        return dumb_r(n_timesteps, n_freqs, n_cats)
    if model_name == 'dumb_dense_tall':
        return dumb_dense_tall(n_timesteps, n_freqs, n_cats)
    if model_name == 'dumb_r_tall':
        return dumb_r_tall(n_timesteps, n_freqs, n_cats)
    if model_name == 'dummy':
        return dummy_model(n_timesteps, n_freqs, n_cats)
    if model_name == 'm800msx256':
        return m800msx256(n_timesteps, n_freqs, n_cats)
    if model_name == 'm800msxf512':
        return m800msxf512(n_timesteps, n_freqs, n_cats)
    else:
        raise KeyError("could not find model {}".format(model_name))

def dummy_model(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(1,
                     kernel_size=(1, 1),
                     activation='relu',
                     input_shape=(n_freqs, n_timesteps, 1)))
    model.add(Flatten())
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def recurrent(n_cats, batch_size):
    a = Input(batch_shape=(batch_size, 1, n_cats))
    x = GRU(50, dropout=0.1, return_sequences=True, stateful=True, recurrent_dropout=0.1)(a)
    x = GRU(50, dropout=0.1, stateful=True, recurrent_dropout=0.1)(x)
    b = Dense(n_cats, activation='softmax')(x)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def okanoya2(n_freqs, n_timesteps, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    a = Input(shape=(n_freqs, n_timesteps, 1))
    x = Conv2D(16, kernel_size=(4, 4), activation='relu')(a)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)
    x = Conv2D(16, kernel_size=(4, 4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(100, kernel_size=(4, 4), activation='relu')(x)
    x = Flatten((2, 1, 3))(x)
    x = GRU(100, activation='relu', dropout=0.1,)(x)
    b = Dense(n_cats, activation='softmax')(x)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01,
                                                 decay=1e-6,
                                                 momentum=0.9,
                                                 nesterov=True),
                  metrics=['accuracy'])
    #model.add(Reshape((1,-1)))
    #model.add(GRU(100, activation='relu', dropout=0.1, stateful=True,
    #    batch_size=batch_input_shape))
    return model

def dumb_dense_tall(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 13), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
def dumb_r(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(Reshape((4, -1)))
    model.add(GRU(50, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def dumb_r_tall(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(Reshape((6, -1)))
    model.add(GRU(50, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

def m800msx256(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    #model.add(Flatten())
    model.add(TimeDistributed(Flatten()))
    #model.add(Reshape((7, -1)))
    model.add(Bidirectional(GRU(32)))
    #model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01,
                                                 decay=1e-6,
                                                 momentum=0.9,
                                                 nesterov=True),
                  metrics=['accuracy'])
    return model

def m800msxf512(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 13), activation='relu'))
    model.add(Reshape((26, -1)))
    model.add(GRU(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def dumb_r(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(Reshape((4, -1)))
    model.add(GRU(50, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def okanoya_r_model(n_freqs, n_timesteps, n_cats):
    # this model has time as first axis
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(n_freqs, n_timesteps, 1))))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 1))))
    model.add(TimeDistributed(Conv3D(16, kernel_size=(4, 4), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv3D(100, kernel_size=(4, 4), activation='relu')))
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(50, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
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
                     input_shape=(n_timesteps, n_freqs, 1)))
#    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))  # CCCP
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2)))
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
