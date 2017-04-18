import cat2csv
import bark
from glob import glob
import numpy as np
from os.path import join, splitext, basename
import resin


def default_image_directory(modelparams, birdparams):
    '''Builds a default place to save the test images based on
    the two input parameter file names.'''
    return ('{}_{}'.format(
        splitext(modelparams)[0], splitext(basename(birdparams))[0]))


def default_model_filename(modelparams, birdparams):
    '''Builds a default place to save the model based on
    the two input parameter file names.'''
    return default_image_directory(modelparams, birdparams) + '.h5'


def model_dims(params):
    'returns a dimension tuple for a model for given parameters'
    p = params
    in_dim1 = p['window_len'] * 2 + 1
    in_dim2 = len(create_spectra(p)._freqs)
    outshape = len(p['encoder']) + 1
    return in_dim1, in_dim2, outshape


def decode(encoder):
    ''' inverts an encoder lookup table
    also adds and entry for silence (0), whose value is '_'.
    example: encoder = {'a': 1, 'b': 2}
             decoder = {0: '_', 1: 'a', 2: 'b'}
    '''
    decoder = {v: k for k, v in encoder.items()}
    decoder[0] = '_'
    return decoder


def create_spectra(params):
    'Creates an interable resin spectra object from a parameters file.'
    p = params
    noverlap = p['NFFT'] - p['window_spacing']
    spa = resin.ISpectra(rate=p['sr'],
                         freq_range=p['freq_range'],
                         n_tapers=p['n_tapers'],
                         NFFT=p['NFFT'],
                         data_window=p['data_window'],
                         noverlap=noverlap)
    window_len = p['window_len']
    n_timesteps = window_len * 2 + 1
    print("time span: ", p['window_spacing'] / p['sr'] * n_timesteps, 's')
    print("freq span: ", spa._freqs[0], 'Hz to ', spa._freqs[-1], 'Hz')
    print("model input dimensions: ", n_timesteps, len(spa._freqs))
    return spa


def read_files(bird_dir, load_events):
    '''
    bird_dir: location of data
    load_events: If true, also load matching csvs

    Reads raw files for testing and training.

    Returns a list of sampled datasets and a list of event datasets
    '''
    data_files = glob(join(bird_dir, "*.dat"))
    print('number of files: ', len(data_files))
    sampled_dsets = [bark.read_sampled(dfile) for dfile in data_files]
    if not load_events:
        return sampled_dsets
    target_files = [splitext(x)[0] + ".csv" for x in data_files]
    event_dsets = [bark.read_events(tfile) for tfile in target_files]
    return sampled_dsets, event_dsets


def save(y_est, basename, p, y_true=None):
    '''Saves model predictions (y_est or y_hat)
    and optionally, the known true targets (y_true)
    '''
    m = basename
    decoder = {v: k for k, v in p['encoder'].items()}
    sampling_rate = 1 / (p['window_spacing'] / p['sr'])
    cat2csv.main(y_est, sampling_rate, decoder, m + '_yhat.csv')
    np.savez(m + '_yhat.npz', yhat=y_est)
    if y_true is not None:
        np.savez(m + '_y_yhat.npz', y=y_true, yhat=y_est)
        bark.write_sampled(m + '_y_yhat.dat',
                np.column_stack([(y_true * 256).astype('int16'),
                                 (y_est * 256).astype('int16')]),
                sampling_rate)
