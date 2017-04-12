from os.path import splitext, join
import bark
import resin
from preprocess import image_iterator, all_data_generator
from model import get_model
import yaml
from utils import read_files, create_spectra


def all_train(model, spa, params):
    sampled_dsets, event_dsets = read_files(params, 'train')
    batch_size = params['batch_size']
    total_steps = 0
    for sampled_dset in sampled_dsets:
        n_specs = len(sampled_dset.data) / (spa._NFFT - spa._noverlap)
        total_steps += n_specs / batch_size
    data_gen = all_data_generator(spa, sampled_dsets, event_dsets, params)
    model.fit_generator(data_gen,
                        total_steps,
                        epochs=params['n_epochs'],
                        verbose=1)

def train_from_images(model, params):
    ''' trains network on prebuilt image dataset'''
    p = params
    im_gen = image_iterator(join(p['image_dir'], p['bird']),
            p['batch_size'],
            p['encoder'])
    next(im_gen)
    model.fit_generator(im_gen, 1000, epochs=params['n_epochs'], verbose=1) 


def save(model, params):
    basename = join(params['model_dir'], params['model'])
    modelfname = '{}_{}_{}.h5'.format(basename, params['model_ver'],
                                      params['bird'])
    model.save(modelfname)


def main(paramfile, raw):
    p = yaml.safe_load(open(paramfile, 'r'))
    # spectral parameters
    spa = create_spectra(p)
    window_len = p['window_len']
    n_timesteps = window_len * 2 + 1
    # get model
    model = get_model(p['model'],
                      len(spa._freqs),
                      n_timesteps,
                      n_cats=len(p['encoder']) + 1)
    # train!
    if raw:
        all_train(model, spa, p)
    else:
        train_from_images(model, p)

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
    p.add_argument('-r', '--raw',
            help='''compute ffts on the fly, 
            otherwise uses images created by to_images.py''',
            action='store_true')
    args = p.parse_args()
    main(args.params, args.raw)
