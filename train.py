from model import get_model
from utils import model_dims, default_model_filename, default_image_directory
from preprocess import image_iterator
import yaml
import os.path


def train_from_images(model, image_dir, params):
    ''' trains network on prebuilt image dataset'''
    im_gen = image_iterator(image_dir, params['batch_size'], params['encoder'])
    model.fit_generator(im_gen, params['steps_per_epoch'], epochs=params['epochs'], verbose=1)


def main(modelparams, birdparams, modelfilename=None, imagedir=None):
    p = yaml.safe_load(open(modelparams, 'r'))
    p.update(yaml.safe_load(open(birdparams, 'r')))
    if imagedir is None:
        imagedir = default_image_directory(modelparams, birdparams)
    imagedir = os.path.join(imagedir, 'train')
    m = get_model(p['model'], *model_dims(p))
    train_from_images(m, imagedir, p)
    if modelfilename is None:
        modelfilename = default_model_filename(modelparams, birdparams)
    m.save(modelfilename)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="""
    trains a neural network
    sampled data must have the extension .dat
    labels must have the same name as data, but with the .csv extension
    """)
    p.add_argument('params', help='a parameters file')
    p.add_argument('birdparams', help='bird-specific parameters')
    p.add_argument('--imagedir', help='directory containing training examples')
    p.add_argument('-o', '--out', help='location to save trained model')
    args = p.parse_args()
    main(args.params, args.birdparams, args.out, args.imagedir) 
