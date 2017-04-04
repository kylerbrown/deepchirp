import numpy as np


def get_encoder_and_decoder(labels):
    encoder = {x: i + 1 for i, x in enumerate(labels.name.unique())}
    decoder = {v: k for k, v in encoder.items()}
    return encoder, decoder


def target_from_events(labels, encoder, t):
    '''
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
        x_sample: a feature vector
        t: the time of the center sample
    '''
    n_freqs = len(spa._freqs)
    power_spec = spa.signal(data).power()
    spec_sr = 1/ ((spa._NFFT - spa._noverlap) / spa._rate)
    # for an x_sample, the first dimension is frequency, second is time
    x_sample = np.zeros((n_freqs, window_len * 2 + 1, 1), dtype='float32')
    i = 0
    for pxx, t in power_spec:
        # shift everything back one time window
        x_sample = np.roll(x_sample, -1, 1)
        # add new power spectra to last time step
        x_sample[:, -1, 0] = np.sqrt(pxx) / amplitude_norm
        if i > window_len:
            yield x_sample, t - (window_len + 1) / spec_sr
        i += 1

    for _ in range(window_len + 1):
        x_sample = np.roll(x_sample, -1, 1)
        # keep yielding, adding a pad
        x_sample[:, -1, 0] = 0
        t += 1 / spec_sr
        yield x_sample, t - (window_len + 1) / spec_sr


def data_generator(spa,
                   data,
                   window_len,
                   labels=None,
                   encoder=None,
                   batch_size=32,
                   amplitude_norm=1):
    '''
    spa       : a resin.Spectra instance
    data      : the raw timeseries data
    labels    : event data with known labels
    encoder   : dictionary mapping labels to numbers
    window_len : the number power spectra in one side of the window. Full window is window_len * 2 + 1.
    spec_sr : spectrogram sampling rate, inverse of the spacing between spectra
    '''
    batch_features = []
    batch_targets = []
    for x, t in windowed_sample_iterator(spa, data, window_len, amplitude_norm):
        batch_features.append(x)
        if labels is not None:
            batch_targets.append(target_from_events(labels, encoder, t))
        if len(batch_features) == batch_size:
            if labels is not None:
                yield np.array(batch_features), np.array(batch_targets)
            else:
                yield np.array(batch_features)
            batch_features = []
            batch_targets = []
    if batch_features:
        if labels is not None:
            yield np.array(batch_features), np.array(batch_targets)
        else:
            yield np.array(batch_features)
