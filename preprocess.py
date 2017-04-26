import os
from glob import iglob
import numpy as np
from keras.utils import to_categorical


def get_encoder_and_decoder(labels):
    encoder = {x: i + 1 for i, x in enumerate(labels.name.unique())}
    decoder = {v: k for k, v in encoder.items()}
    return encoder, decoder

def all_targets_from_events(events_df, n_targets, fft_interval, encoder, sr):
    labels = events_df.to_dict('records')
    t = np.arange(0, n_targets * fft_interval, fft_interval)
    # format target data from events
    targets = np.zeros_like(t)
    for syl in labels:
        name, start, stop = syl['name'], syl['start'], syl['stop']
        start_samp = int((start * sr) / fft_interval)
        stop_samp = int((stop * sr) / fft_interval)
        if name in encoder:
            targets[start_samp: stop_samp] = encoder[name]
    return to_categorical(targets, num_classes=len(encoder)+1)


def encode_targets(targets, encoder):
    ''' Coverts a list of targets as strings to one-hot encoded 
    categoricals. Target is 0 if not in encoder
    
    encoder - a dictionary with string keys and integer values'''

    enc_targets = [encoder[t] if t in encoder else 0 for t in targets]
    return to_categorical(enc_targets, num_classes=len(encoder) + 1)


def count_image_samples(directory):
    '''gets the number of samples within subdirectories
    useful for knowing how many samples to retrieve from the 
    test iterator'''
    classes = [o for o in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, o))]
    n_samples = [sum(1 for _ in iglob(os.path.join(directory, k, '*.npy'))) for k in classes]
    return sum(n_samples)

def get_classes(directory):
    '''Returns three lists:
    classes : a list of the names of each subdirectory
    subdirs : the names of the subdirectories
    n_samples : the number of samples in each subdirectory
    '''
    classes = [o for o in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, o))]
    subdirs = [os.path.join(directory, c) for c in classes]
    n_samples = [sum(1 for _ in iglob(os.path.join(directory, k, '*.npy'))) for k in classes]
    return classes, subdirs, n_samples

def test_image_iterator(directory, batch_size, encoder, loop=True):
    ''' Samples sequentially from images in subdirectories
    subdirectory name is the target name

    If subdirectory name is not in encoder, target is assumed to be
    the first class (0).
    '''
    classes, subdirs, n_samples = get_classes(directory)
    n_classes = len(classes)
    counter = [0 for _ in classes]
    print('number of test examples', list(zip(classes, n_samples)))
    do_loop = True
    while do_loop:
        xs = []
        ys = []
        for ith_class in range(len(classes)):
            for j in range(n_samples[ith_class]):
                xfile = os.path.join(subdirs[ith_class],
                        '{:06}.npy'.format(j))
                x = np.load(xfile)
                x = np.expand_dims(x, 2)  # keras expects third dimension
                xs.append(x)
                ys.append(classes[ith_class])
                if len(xs) >= batch_size:
                    x_batch = np.array(xs)
                    y_batch = encode_targets(ys, encoder)
                    yield x_batch, y_batch
                    xs = []
                    ys = []
            x_batch = np.array(xs)
            y_batch = encode_targets(ys, encoder)
            yield x_batch, y_batch
            xs = []
            ys = []
        do_loop = loop

def image_iterator(directory, batch_size, encoder):
    ''' Samples evenly from images in subdirectories
    subdirectory name is the target name

    If subdirectory name is not in encoder, target is assumed to be
    the first class (0).
    '''
    classes, subdirs, n_samples = get_classes(directory)
    n_classes = len(classes)
    counter = [0 for _ in classes]
    sample_permute = [np.random.permutation(range(n)) for n in n_samples]
    print('number of training examples', list(zip(classes, n_samples)))
    ith_class = 0
    while True:
        xs = []
        ys = []
        for i in range(batch_size):
            xfile = os.path.join(subdirs[ith_class],
                    '{:06}.npy'.format(sample_permute[ith_class][counter[ith_class]]))
            x = np.load(xfile)
            x = np.expand_dims(x, 2)  # keras expects third dimension
            xs.append(x)
            ys.append(classes[ith_class])
            counter[ith_class] = (counter[ith_class] + 1) % n_samples[ith_class]
            ith_class = (ith_class + 1) % n_classes
        x_batch = np.array(xs)
        y_batch = encode_targets(ys, encoder)
        yield x_batch, y_batch



def target_from_events(labels, encoder, t):
    '''
    NOTE: SLOW
    labels   : a pandas dataframe containing the columns 'start', 'stop' and 'name'
    encoder   : a dictionary mapping labels to numbers
    returns a single sample target for a single time t '''
    syl = labels[(t >= labels['start']) & (t < labels['stop'])]
    target = np.zeros(len(encoder) + 1, dtype=bool)
    if len(syl) > 0:
        syl_str = syl.name.iloc[0].lower()[0]
        if syl_str in encoder:
            cat = encoder[syl_str]
        else:
            cat = 0
    else:
        cat = 0
    target[cat] = True
    return target

def windowed_sample_iterator(spa, data, window_len, amplitude_norm=1):
    '''
    Collects power spectra into larger chunks to create an input feature of size
    n_freqs X window_len * 2 + 1 X 1.
    Last unit dimension seems required by keras.
    Boundaries are zero-padded.

    yields:
        x_sample: a feature vector dimension: time X frequency
        t: the time of the center sample
    '''
    n_freqs = len(spa._freqs)
    power_spec = spa.signal(data).power()
    spec_sr = 1/ ((spa._NFFT - spa._noverlap) / spa._rate)
    # for an x_sample, the first dimension is frequency, second is time
    x_sample = np.zeros((window_len * 2 + 1, n_freqs, 1), dtype='float32')
    i = 0
    for pxx, t in power_spec:
        # shift everything back one time window
        x_sample = np.roll(x_sample, -1, 0)
        # add new power spectra to last time step
        x_sample[-1, :, 0] = np.sqrt(pxx) / amplitude_norm
        if i > window_len:
            yield x_sample, t - (window_len + 1) / spec_sr
        i += 1

    for _ in range(window_len + 1):
        x_sample = np.roll(x_sample, -1, 0)
        # keep yielding, adding a pad
        x_sample[-1, :, 0] = 0
        t += 1 / spec_sr
        yield x_sample, t - (window_len + 1) / spec_sr


def all_data_generator(spa,
                   sampled_dsets,
                   event_dsets,
                   params):
    while True:
        for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
            data_gen = data_generator(spa,
                                      sampled_dset.data,
                                      window_len=params['window_len'],
                                      labels=event_dset.data,
                                      encoder=params['encoder'],
                                      batch_size=params['batch_size'],
                                      amplitude_norm=params['amplitude_norm'],
                                      loop=False)
            yield from data_gen

def data_generator(spa,
                   data,
                   window_len,
                   labels=None,
                   encoder=None,
                   batch_size=32,
                   amplitude_norm=1,
                   loop=False):
    '''
    spa       : a resin.Spectra instance
    data      : the raw timeseries data
    labels    : event data with known labels
    encoder   : dictionary mapping labels to numbers
    window_len : the number power spectra in one side of the window. Full window is window_len * 2 + 1.
    spec_sr : spectrogram sampling rate, inverse of the spacing between spectra
    '''
    should_loop = True
    while should_loop:
        batch_features = []
        if labels is not None:
            fft_interval = spa._NFFT - spa._noverlap
            n_targets = np.ceil(len(data) / fft_interval)
            targets = all_targets_from_events(labels, n_targets, fft_interval, encoder, spa._rate)
        for i, (x, t) in enumerate(windowed_sample_iterator(spa, data, window_len, amplitude_norm)):
            batch_features.append(x)
            if len(batch_features) == batch_size:
                if labels is not None:
                    yield np.array(batch_features), targets[i - batch_size + 1: i + 1]
                else:
                    yield np.array(batch_features)
                batch_features = []
        # send a final, partially full batch
        if batch_features:
            if labels is not None:
                yield np.array(batch_features), targets[-len(batch_features):]
            else:
                yield np.array(batch_features)
        if loop:
            print("warning, final batch sent, looping...")
            should_loop = True
        else:
            should_loop = False
