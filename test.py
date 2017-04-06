from sklearn.metrics import accuracy_score, classification_report
import keras
from sys import stdout, argv
from glob import glob
from os.path import join, splitext
import pandas as pd
import bark
import resin
from preprocess import all_targets_from_events, data_generator
import numpy as np
import yaml
import cat2csv
from train import read_files, create_spectra

def load_model(params):
    basename = join(params['model_dir'], params['model'])
    modelfname = '{}_{}_{}.h5'.format(basename, params['model_ver'],
                                      params['bird'])
    print(modelfname)
    return keras.models.load_model(modelfname)

def save(y_est, y_true, basename, p):
    m = basename
    decoder = {v: k for k, v in p['encoder'].items()}
    np.savez(m + '_y_yhat.npz', y=y_true, yhat=y_est)
    sampling_rate = 1 / (p['window_spacing'] / p['sr'])
    bark.write_sampled(m + '_y_yhat.dat',
            np.column_stack((y_true.astype('int16') * 256,
                            (y_est * 256).astype('int16'))),
            p['sr'])
    bark.write_sampled(m + '_yhat.dat',
                            y_est,
                            p['sr'], decoder=decoder)
    cat2csv.main(m + '_yhat.dat', p['model'] + '_yhat.csv')


def test(model, sampled_dset, event_dset, spa, p):
    #if len(event_dset.data) == 0:
    #    continue
    data_gen = data_generator(spa, sampled_dset.data, window_len=p['window_len'],
                              labels=event_dset.data, encoder=p['encoder'],
                              batch_size=p['batch_size'], amplitude_norm=p['amplitude_norm'],
                              loop=True)
    n_step_size =  spa._NFFT - spa._noverlap
    n_targets = np.ceil(len(sampled_dset.data) / n_step_size)
    n_steps = np.ceil(n_targets / p['batch_size'])
    print(n_steps)
    print(sampled_dset.name)
    y_est = model.predict_generator(data_gen, n_steps, verbose=1)
    y_true = all_targets_from_events(event_dset.data, n_targets, n_step_size,
            p['encoder'], spa._rate)
    print("accuracy score:", accuracy_score(np.argmax(y_true, 1), np.argmax(y_est, 1)))
    print(classification_report(np.argmax(y_true, 1), np.argmax(y_est, 1)))
    # save outputs
    basename = splitext(sampled_dset.name)[0] 
    save(y_est, y_true, basename, p)

def main(parameters_file):
    p = yaml.safe_load(open(parameters_file, 'r'))
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
    model = load_model(p)
    print(model.summary())
    # test!
    for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
        test(model, sampled_dset, event_dset, spa, p)

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
