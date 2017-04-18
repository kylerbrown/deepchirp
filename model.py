import keras
from keras.models import Sequential, Model
from keras.layers import Permute, Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, GRU, GlobalAveragePooling1D, GlobalAveragePooling2D, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout

def dummy_model(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(1,
                     kernel_size=(1, 1),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(Flatten())
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model



def inception3(n_timesteps, n_freqs, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    a = Input(shape=(n_timesteps, n_freqs, 1))
    x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(a)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(a)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    # pool
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # inception layer
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    b = Dense(n_cats, activation='softmax')(x)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def inception2(n_timesteps, n_freqs, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    a = Input(shape=(n_timesteps, n_freqs, 1))
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(a)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(64, kernel_size=(4, 4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(4, 4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    b = Dense(n_cats, activation='softmax')(x)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def inception(n_timesteps, n_freqs, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    a = Input(shape=(n_timesteps, n_freqs, 1))
    x = Conv2D(32, kernel_size=(5, 5), activation='relu')(a)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, kernel_size=(4, 4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    x = TimeDistributed(GlobalAveragePooling1D())(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(32))(x)
    x = Dropout(0.2)(x)
    b = Dense(n_cats, activation='softmax')(x)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def okanoya2(n_timesteps, n_freqs, n_cats):
    '''n_freqs: length of frequency dimension
    n_timesteps: length of time dimension
    n_cats: number of output categories (number of syllables + 1)
    '''
    a = Input(shape=(n_freqs, n_timesteps, 1))
    x = Conv2D(16, kernel_size=(5, 5), activation='relu')(a)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
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

def dumb_dense_tall(n_timesteps, n_freqs, n_cats):
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

def dumb_r_tall(n_timesteps, n_freqs, n_cats):
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


def simplecnn3(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def simplecnn4(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
def simplecnn2(n_freqs, n_timesteps, n_cats):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.4))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def simplecnn(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def lstm(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(16,
                     kernel_size=(3, 3),
                     strides=(1,2),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def m800msx256(n_timesteps, n_freqs, n_cats):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(n_timesteps, n_freqs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(1, 5), activation='relu'))
    #model.add(Flatten())
    model.add(TimeDistributed(Flatten()))
    #model.add(Reshape((7, -1)))
    model.add(Bidirectional(GRU(32)))
    #model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def m800msxf512(n_timesteps, n_freqs, n_cats):
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

def get_model(modelname, input_size1, input_size2, output_size, *args, **kwargs):
    if modelname not in globals():
        raise KeyError('{} not a function in model.py'.format(modelname))
    return globals()[modelname](input_size1, input_size2, output_size, *args, **kwargs)
