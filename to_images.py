from preprocess import data_generator
import yaml
import numpy as np
from utils import decode, create_spectra, read_files
import os.path

def sampling_encoder(params):
    'add extras to encoder to ensure even sampling'
    encoder = params['encoder'].copy()
    encoder.update(params['sampling_encoder_extras'])
    return encoder

def save_to_image(sampled_dset, event_dset, spa, p):
    encoder = sampling_encoder(p)
    decoder = decode(encoder)
    dg = data_generator(spa,
                   sampled_dset.data,
                   p['window_len'],
                   labels=event_dset.data,
                   encoder=encoder,
                   batch_size=1,
                   amplitude_norm=1,
                   loop=False)
    for i, (x, y) in enumerate(dg):
        yname = decoder[int(np.argmax(y))]
        im = Image.fromarray(np.squeeze((x/np.max(x))*255).astype('uint8').T)
        im.save('images/{}_{:05}_{}.png'.format(sampled_dset.name, i, yname))

def get_generator(spa, params):
    encoder = sampling_encoder(params)
    sampled_dsets, event_dsets = read_files(params, 'train')
    for sampled_dset, event_dset in zip(sampled_dsets, event_dsets):
        data_gen = data_generator(spa,
                                  sampled_dset.data,
                                  window_len=params['window_len'],
                                  labels=event_dset.data,
                                  encoder=encoder,
                                  batch_size=1,  # this is required for saving images
                                  amplitude_norm=params['amplitude_norm'],
                                  loop=False)
        yield from data_gen


def save_to_disk(dg, spa, p, png=False):
    if png:
        from PIL import Image
    encoder = sampling_encoder(p)
    decoder = decode(encoder)
    print('decoder:', decoder)
    counter = {k: 0 for k in decoder.values()}
    for v in decoder.values():
        directory = os.path.join(p['image_dir'], p['bird'], v)
        if not os.path.exists(directory):
            os.makedirs(directory)
    for x, y in dg:
        yname = decoder[int(np.argmax(y))]
        filename = '{:06}'.format(counter[yname])
        fullname = os.path.join(p['image_dir'], p['bird'], yname, filename)
        if png:
            im = Image.fromarray(np.squeeze((x/np.max(x))*255).astype('uint8').T)
            im.save(fullname + '.png')
        else:
            np.save(fullname + '.npy', np.squeeze(x))
        counter[yname] += 1


def main(paramfile, png):
    p = yaml.safe_load(open(paramfile, 'r'))
    spa = create_spectra(p)
    datagen = get_generator(spa, p)
    save_to_disk(datagen, spa, p, png)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='''
    Create a series of labeled two-dimensional 'images' from a training dataset
    see what the neural network sees!

    sampled data must have the extension .dat
    labels must have the same name as data, but with the .csv extension

    by default, the program outputs numpy arrays to disk, but you can also output
    PNG images. Only numpy arrays can be used for training, but the PNG images
    are useful for visualizing the input.
    ''')
    p.add_argument('params', help='a parameters file')
    p.add_argument('--png', help='save output as viewable images',
                   action='store_true')
    args = p.parse_args()
    main(args.params, args.png)
