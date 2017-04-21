from glob import iglob
from preprocess import data_generator
import yaml
import numpy as np
import os.path
import shutil
from utils import decode, create_spectra, read_files, default_image_directory

def sampling_encoder(params):
    'add extras to encoder to ensure even sampling'
    encoder = params['encoder'].copy()
    extras = {x: len(encoder) + 1 + ith
            for ith, x in enumerate(params['sampling_encoder_extras'])}
    encoder.update(extras)
    return encoder

def get_generator(trainingdir, spa, params):
    encoder = sampling_encoder(params)
    sampled_dsets, event_dsets = read_files(trainingdir, params)
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

def set_aside_test_fraction(train_dir, test_dir, test_fraction, png):
    '''from each training example type, set aside the last test_fraction
    for testing
    
    train_dir: directory with training examples as subdirectories
    test_dir: location to place test images
    test_fraction: fraction of examples to move. eg 0.01 will move 1% of samples
                    from train_dir to test_dir
    '''
    ext = '.png' if png else '.npy'
    classes = [o for o in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, o))]
    n_classes = len(classes)
    subdirs = [os.path.join(train_dir, c) for c in classes]
    test_subdirs = [os.path.join(test_dir, c) for c in classes]
    counter = [0 for _ in classes]
    n_samples = [sum(1 for _ in iglob(os.path.join(train_dir, k, '*.npy'))) for k in classes]
    n_test = [int(n_samp * test_fraction) for n_samp in n_samples]
    for directory in test_subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    for ith_class, c in enumerate(classes):
        train_offset = n_samples[ith_class] - n_test[ith_class]
        for i in range(n_test[ith_class]):
            oldname = os.path.join(train_dir, c, 
                    '{:06}{}'.format(i + train_offset, ext))
            newname = os.path.join(test_dir, c, 
                    '{:06}{}'.format(i, ext))
            shutil.move(oldname, newname)


def save_to_disk(image_dir, dg, spa, p, png=False, max_samples_per_category=50000):
    if png:
        from PIL import Image
    encoder = sampling_encoder(p)
    decoder = decode(encoder)
    print('decoder:', decoder)
    train_counter = {k: 0 for k in decoder.values()}
    counter = {k: 0 for k in decoder.values()}
    for v in decoder.values():
        directory = os.path.join(image_dir, v)
        if not os.path.exists(directory):
            os.makedirs(directory)
    for x, y in dg:
        yname = decoder[int(np.argmax(y))]
        if counter[yname] > max_samples_per_category:
            continue
        filename = '{:06}'.format(counter[yname])
        fullname = os.path.join(image_dir, yname, filename)
        if png:
            im = Image.fromarray(np.squeeze((x/np.max(x))*255).astype('uint8').T)
            im.save(fullname + '.png')
        else:
            np.save(fullname + '.npy', np.squeeze(x))
        counter[yname] += 1


def main(trainingdir, paramfile, birdparams,
        png=False,
        outdir=None,
        test_fraction=0.05):
    p = yaml.safe_load(open(paramfile, 'r'))
    p.update(yaml.safe_load(open(birdparams, 'r')))
    spa = create_spectra(p)
    datagen = get_generator(trainingdir, spa, p)
    if outdir is None:
        outdir = default_image_directory(paramfile, birdparams)
    out_train_dir = os.path.join(outdir, 'train')
    out_test_dir = os.path.join(outdir, 'test')
    save_to_disk(out_train_dir, datagen, spa, p, png, p['max_samples_per_category'])
    set_aside_test_fraction(out_train_dir, out_test_dir, test_fraction, png)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='''
    Create a series of labeled two-dimensional 'images' from a training dataset

    sampled data must have the extension .dat
    labels must have the same name as data, but with the .csv extension

    both .dat and .csv files must have Bark metadata files.

    By default, the program outputs numpy arrays to disk, but you can also output
    PNG images. Only numpy arrays can be used for training, but the PNG images
    are useful for visualizing the input.
    ''')
    p.add_argument('trainingdir', help='directory containing training examples')
    p.add_argument('params', help='a parameters file')
    p.add_argument('birdparams', help='bird-specific parameters')
    p.add_argument('-o', '--outdir', help='directory to save training images')
    p.add_argument('-t', '--testfraction', help='fraction of images to save for testing, default=0.05',
            type=float)
    p.add_argument('--png', help='save output as viewable images',
                   action='store_true')
    args = p.parse_args()
    main(args.trainingdir, args.params, args.birdparams, args.png, args.outdir)
