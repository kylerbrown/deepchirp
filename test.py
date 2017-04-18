from preprocess import all_targets_from_events, data_generator, test_image_iterator
from utils import read_files, create_spectra, default_model_filename, save, default_image_directory
import os.path
import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import yaml

def test_images(model, image_dir, p):
    image_gen = test_image_iterator(image_dir, p['batch_size'], encoder=p['encoder'],
            loop=False)
    y = []
    yhat = []
    for batch_x, batch_y in image_gen:
        batch_yhat = model.predict_on_batch(batch_x)
        y.append(batch_y)
        yhat.append(batch_yhat)
    
    yhat = np.row_stack(yhat)
    y = np.row_stack(y)
    print("accuracy score:", accuracy_score(np.argmax(y, 1), np.argmax(yhat, 1)))
    print(classification_report(np.argmax(y, 1), np.argmax(yhat, 1)))

def test(model, sampled_dset, event_dset, spa, p):
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
    basename = '{}_{}'.format(os.path.splitext(sampled_dset.path)[0], p['model'])
    save(y_est, basename, p, y_true)

def main(parameters_file, bird_params, modelfile=None, imagedir=None):
    p = yaml.safe_load(open(parameters_file, 'r'))
    p.update(yaml.safe_load(open(bird_params, 'r')))
    if imagedir is None:
        imagedir = default_image_directory(parameters_file, bird_params)
    imagedir = os.path.join(imagedir, 'test')
    if modelfile is None:
        modelfile = default_model_filename(parameters_file, bird_params)
    model = keras.models.load_model(modelfile)
    print(model.summary())
    test_images(model, imagedir, p)
    #for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
    #    test(model, sampled_dset, event_dset, spa, p)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="""
    trains a neural network
    sampled data must have the extension .dat
    labels must have the same name as data, but with the .csv extension
    """)
    p.add_argument('params', help='a model parameters file')
    p.add_argument('birdparams', help='a bird parameters file')
    p.add_argument('--imagedir', help='directory containing training examples')
    p.add_argument('-m', '--model', help='model to test')
    args = p.parse_args()
    main(args.params, args.birdparams, args.model, args.imagedir)
