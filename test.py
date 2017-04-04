from sklearn.metrics import accuracy_score, classification_report
import keras
from sys import stdout, argv
from glob import glob
from os.path import splitext
import pandas as pd
import bark
import resin
from preprocess import get_encoder_and_decoder, data_generator
import numpy as np
import yaml

parameters_file = argv[1]

p = yaml.safe_load(open(parameters_file, 'r'))
noverlap=p['NFFT']-p['window_spacing']
n_timesteps = p['window_len'] * 2 + 1
spa = resin.ISpectra(rate=p['sr'], freq_range=p['freq_range'], n_tapers=p['n_tapers'], NFFT=p['NFFT'],
                    data_window=p['data_window'], noverlap=noverlap)
test_data_files = glob(p['TEST_DIR'] + '*.dat')
test_target_files = [splitext(x)[0] + '.csv' for x in test_data_files]
print('test files: ', len(test_data_files))

sampled_dsets = [bark.read_sampled(dfile) for dfile in test_data_files]
event_dsets = [bark.read_events(tfile) for tfile in test_target_files]
model = keras.models.load_model(p['modelfile'])

for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
    if len(event_dset.data) == 0:
        continue
    data_gen = data_generator(spa, sampled_dset.data, window_len=p['window_len'],
                              labels=event_dset.data, encoder=p['encoder'],
                              batch_size=p['batch_size'], amplitude_norm=p['amplitude_norm'])
    n_step_size =  (spa._NFFT - spa._noverlap)
    n_steps = len(sampled_dset.data) // n_step_size
    print(sampled_dset.name)
    y_true = []
    y_est = []
    for i, (x_train, y_train) in enumerate(data_gen):
        y_pred = model.predict_proba(x_train, batch_size=len(x_train))
        y_true.append(y_train)
        y_est.append(y_pred)
y_true = np.row_stack(y_true)
y_est = np.row_stack(y_est)

print("accuracy score:", accuracy_score(np.argmax(y_true, 1), np.argmax(y_est, 1)))
print(classification_report(np.argmax(y_true, 1), np.argmax(y_est, 1)))
np.savez(p['test_data'], y=y_true, yhat=y_est)
