import keras
from sys import stdout, argv
from glob import glob
from os.path import splitext
import bark
import resin
from preprocess import data_generator
import numpy as np
import yaml
from model import get_model

parameters_file = argv[1]

p = yaml.safe_load(open(parameters_file, 'r'))
noverlap=p['NFFT']-p['window_spacing']
n_timesteps = p['window_len'] * 2 + 1
spa = resin.ISpectra(rate=p['sr'], freq_range=p['freq_range'], n_tapers=p['n_tapers'], NFFT=p['NFFT'],
                    data_window=p['data_window'], noverlap=noverlap)
test_data_files = glob(p['TRAIN_DIR'] + '*.dat')
test_target_files = [splitext(x)[0] + '.csv' for x in test_data_files]
print('test files: ', len(test_data_files))

sampled_dsets = [bark.read_sampled(dfile) for dfile in test_data_files]
event_dsets = [bark.read_events(tfile) for tfile in test_target_files]
model = keras.models.load_model(p['modelfile'])
encoder = p['encoder']
model2 = get_model(p['sequencemodel'], len(spa._freqs), n_timesteps, n_cats=len(encoder) + 1,
        batch_size=p['batch_size'])

for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
    if len(event_dset.data) == 0:
        continue
    for ep in range(p['n_epochs_recurrent']):
        data_gen = data_generator(spa, sampled_dset.data, window_len=p['window_len'],
                                  labels=event_dset.data, encoder=p['encoder'],
                                  batch_size=p['batch_size'], amplitude_norm=p['amplitude_norm'])
        n_step_size =  (spa._NFFT - spa._noverlap)
        n_steps = len(sampled_dset.data) // n_step_size
        print(sampled_dset.name)
        for i, (x_train, y_train) in enumerate(data_gen):
            if len(y_train) != p['batch_size']:
                continue
            m1_output = model.predict_proba(x_train, batch_size=len(x_train))
            m2_input = np.expand_dims(m1_output, axis=1)
            model2.fit(m2_input, y_train,
                       batch_size=len(y_train), epochs=1, shuffle=False)

model2.save(p['sequencemodelfile'])
