import keras
from keras.models import Sequential, Model
from keras.layers import Permute, Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, GRU, GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout

def get_model(model_name, n_timesteps, n_freqs, n_cats, **kwargs):
    print(model_name)
    if model_name == 'recurrent':
        m =  recurrent(n_cats, **kwargs)
    elif model_name == 'okanoya_r':
        m = okanoya_r_model(n_timesteps, n_freqs, n_cats, **kwargs)
    elif model_name == 'okanoya':
        m = okanoya_model(n_timesteps, n_freqs, n_cats)
    elif model_name == 'inception':
        m = inception(n_timesteps, n_freqs, n_cats)
    elif model_name == 'dumb_r':
        m = dumb_r(n_timesteps, n_freqs, n_cats)
    elif model_name == 'dumb_dense_tall':
        m = dumb_dense_tall(n_timesteps, n_freqs, n_cats)
    elif model_name == 'dumb_r_tall':
        m = dumb_r_tall(n_timesteps, n_freqs, n_cats)
    elif model_name == 'dummy':
        m = dummy_model(n_timesteps, n_freqs, n_cats)
    elif model_name == 'm800msx256':
        m = m800msx256(n_timesteps, n_freqs, n_cats)
    elif model_name == 'm800msxf512':
        m = m800msxf512(n_timesteps, n_freqs, n_cats)
    else:
        raise KeyError("could not find model {}".format(model_name))
    print(m.summary())
    return m

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


def inception(n_freqs, n_timesteps, n_cats):
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
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    x = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    # inception layer
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)
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

def okanoya2(n_freqs, n_timesteps, n_cats):
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
