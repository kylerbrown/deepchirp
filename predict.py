from utils import default_model_filename, save, read_files, create_spectra
from preprocess import data_generator
import keras
import numpy as np
import os.path
import yaml


def predict(model, sampled_dset, spa, p):
    # note: loop must be set to true to enable using keras's predict_generator
    data_gen = data_generator(spa,
                              sampled_dset.data,
                              window_len=p['window_len'],
                              encoder=p['encoder'],
                              batch_size=p['predict_batch_size'],
                              amplitude_norm=p['amplitude_norm'],
                              loop=True)
    n_step_size = spa._NFFT - spa._noverlap
    n_targets = np.ceil(len(sampled_dset.data) / n_step_size)
    n_steps = np.ceil(n_targets / p['predict_batch_size'])
    print(n_steps)
    print(sampled_dset.name)
    y_est = model.predict_generator(data_gen, n_steps, verbose=1)
    # save outputs
    basename = '{}_{}'.format(
        os.path.splitext(sampled_dset.path)[0], p['model'])
    save(y_est, basename, p)


def main(parameters_file, bird_params, predict_dir, modelfile=None):
    p = yaml.safe_load(open(parameters_file, 'r'))
    p.update(yaml.safe_load(open(bird_params, 'r')))
    sampled_dsets = read_files(predict_dir, load_events=False)
    spa = create_spectra(p)
    if modelfile is None:
        modelfile = default_model_filename(parameters_file, bird_params)
    model = keras.models.load_model(modelfile)
    print(model.summary())
    for sampled_dset in sampled_dsets:
        predict(model, sampled_dset, spa, p)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="""
    Uses a pretrained neural network to for syllable recognition and classification
    sampled data must have the extension .dat
    """)
    p.add_argument('params', help='a model parameters file')
    p.add_argument('birdparams', help='a bird parameters file')
    p.add_argument('predictdir', help='directory containing data')
    p.add_argument('-m', '--model', help='model to test')
    args = p.parse_args()
    main(args.params, args.birdparams, args.predictdir, args.model)
