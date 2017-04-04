from sys import stdout, argv
from glob import glob
from os.path import splitext
import pandas as pd
import bark
import resin
from preprocess import get_encoder_and_decoder, data_generator
from model import get_model
import numpy as np
import yaml

parameters_file = argv[1]

# sampled data must have the extension .dat
# labels must have the same name as data, but with the .csv extension
p = yaml.safe_load(open(parameters_file, 'r'))
TRAIN_DIR = p['TRAIN_DIR']
# spectral parameters
sr = p['sr']
NFFT = p['NFFT']
data_window= p['data_window']
freq_range = p['freq_range']
n_tapers = p['n_tapers']
window_spacing = p['window_spacing']
window_len = p['window_len']

noverlap=NFFT-window_spacing
n_timesteps = window_len * 2 + 1
print("time_span: ", window_spacing/sr * n_timesteps)
spa = resin.ISpectra(rate=sr, freq_range=freq_range, n_tapers=n_tapers, NFFT=NFFT,
                    data_window=data_window, noverlap=noverlap)

# files
test_data_files = glob(TRAIN_DIR + "*.dat")
test_target_files = [splitext(x)[0] + ".csv" for x in test_data_files]
print('test files: ', len(test_data_files))

sampled_dsets = [bark.read_sampled(dfile) for dfile in test_data_files]
event_dsets = [bark.read_events(tfile) for tfile in test_target_files]

# target encoder
encoder = p['encoder']
class_weights = {v: p['syllable_class_weight'] for v in encoder.values()}
class_weights[0] = 1
print("model input dimensions: ", len(spa._freqs), n_timesteps)
batch_size = p['batch_size']
model = get_model(p['model'], len(spa._freqs), n_timesteps, n_cats=len(encoder) + 1)
n_epochs = p['n_epochs']
print(model.summary())

for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
    if len(event_dset.data) == 0:
        continue
    for epoch in range(n_epochs):
        data_gen = data_generator(spa, sampled_dset.data, window_len=window_len,
                                  labels=event_dset.data, encoder=encoder,
                                  batch_size=batch_size, amplitude_norm=p['amplitude_norm'])
        n_step_size =  (spa._NFFT - spa._noverlap)
        n_steps = len(sampled_dset.data) // n_step_size
        print(sampled_dset.name)
        print("epoch: ", epoch)
        for i, (x_train, y_train) in enumerate(data_gen):
            loss, acc = model.train_on_batch(x_train, y_train, class_weight=class_weights)
            stdout.write('\r')
            stdout.write('{:.2%}\tloss: {:e}\tacc: {:.2%}'.format(i/(n_steps/batch_size), loss, acc))
            stdout.flush()
            print(np.mean(y_train, 0))

# save model
model.save(p['modelfile'])
