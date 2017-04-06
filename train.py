from glob import glob
from os.path import splitext, join
import bark
import resin
from preprocess import data_generator
from model import get_model
import yaml


def create_spectra(params):
    p = params
    noverlap = p['NFFT'] - p['window_spacing']
    spa = resin.ISpectra(rate=p['sr'],
                         freq_range=p['freq_range'],
                         n_tapers=p['n_tapers'],
                         NFFT=p['NFFT'],
                         data_window=p['data_window'],
                         noverlap=noverlap)
    return spa


def read_files(params, kind):
    ''' kind may one of 'train', 'test'
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


def train(model, sampled_dset, event_dset, spa, params):
    class_weights = {v: params['syllable_class_weight']
                     for v in params['encoder'].values()}
    class_weights[0] = 1  # silence is always weighted 1
    batch_size = params['batch_size']
    data_gen = data_generator(spa,
                              sampled_dset.data,
                              window_len=params['window_len'],
                              labels=event_dset.data,
                              encoder=params['encoder'],
                              batch_size=batch_size,
                              amplitude_norm=params['amplitude_norm'],
                              loop=True)
    n_specs = len(sampled_dset.data) / (spa._NFFT - spa._noverlap)
    n_steps = n_specs / batch_size
    model.fit_generator(data_gen, n_steps, verbose=1)


def save(model, params):
    basename = join(params['model_dir'], params['model'])
    modelfname = '{}_{}_{}.h5'.format(basename, params['model_ver'],
                                      params['bird'])
    model.save(modelfname)


def main(paramfile):
    p = yaml.safe_load(open(paramfile, 'r'))
    # load files: inputs and targets
    sampled_dsets, event_dsets = read_files(p, 'test')
    # spectral parameters
    spa = create_spectra(p)
    window_len = p['window_len']
    n_timesteps = window_len * 2 + 1
    print("time span: ", p['window_spacing'] / p['sr'] * n_timesteps)
    print("freq span: ", spa._freqs[0], spa._freqs[-1])
    print("model input dimensions: ", n_timesteps, len(spa._freqs))
    # get model
    model = get_model(p['model'],
                      len(spa._freqs),
                      n_timesteps,
                      n_cats=len(p['encoder']) + 1)
    print(model.summary())
    # train!
    for epoch in range(p['n_epochs']):
        for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
            print(sampled_dset.name)
            print("epoch: ", epoch)
            train(model, sampled_dset, event_dset, spa, p)
    # save results
    save(model, p)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="""
    trains a neural network
    sampled data must have the extension .dat
    labels must have the same name as data, but with the .csv extension
    """)
    p.add_argument('params', help='a parameters file')
    args = p.parse_args()
    main(args.params)
