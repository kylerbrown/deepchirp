from os.path import join, splitext
from glob import glob
import resin
import bark

def decode(encoder):
    ''' inverts an encoder lookup table
    also adds and entry for silence (0), whose value is '_'.
    example: encoder = {'a': 1, 'b': 2}
             decoder = {0: '_', 1: 'a', 2: 'b'}
    '''
    decoder = {v:k for k, v in encoder.items()}
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


def read_files(params, kind):
    ''' 
    Reads raw files for testing and training.
    Loads bark datasets as specified in a parameters file
    kind may one of 'train', 'test'

    returns a list of sampled datasets and a list of event datasets
    '''
    # files
    if kind == 'train':
        dir_ = params['TRAIN_DIR']
    elif kind == 'test':
        dir_ = params['TEST_DIR']
    bird_dir = join(dir_, params['bird'])
    data_files = glob(join(bird_dir, "*.dat"))
    target_files = [splitext(x)[0] + ".csv" for x in data_files]
    print('number of files: ', len(data_files))
    sampled_dsets = [bark.read_sampled(dfile) for dfile in data_files]
    event_dsets = [bark.read_events(tfile) for tfile in target_files]
    return sampled_dsets, event_dsets
