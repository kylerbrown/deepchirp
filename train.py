import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint
from model import get_model
from utils import model_dims, default_model_filename, default_image_directory
from preprocess import test_image_iterator, image_iterator, get_classes
import yaml
import os.path

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def train_from_images(model, image_dir, params, modelfilename):
    ''' trains network on prebuilt image dataset'''
    im_gen = image_iterator(os.path.join(image_dir, 'train'), params['batch_size'], params['encoder'])
    val_dir = os.path.join(image_dir, 'test')
    val_steps = sum(get_classes(val_dir)[2]) // params['batch_size'] + 1
    val_gen = test_image_iterator(val_dir, params['batch_size'],
            params['encoder'], loop=True)
    
    model.fit_generator(im_gen, params['steps_per_epoch'], epochs=params['epochs'], verbose=1,
            validation_data=val_gen, validation_steps=val_steps, 
            callbacks=([ModelCheckpoint(modelfilename, save_best_only=True), TensorBoard()]))


def main(modelparams, birdparams, modelfilename=None, imagedir=None):
    p = yaml.safe_load(open(modelparams, 'r'))
    p.update(yaml.safe_load(open(birdparams, 'r')))
    if imagedir is None:
        imagedir = default_image_directory(modelparams, birdparams)
    m = get_model(p['model'], *model_dims(p))
    print(m.summary())
    if modelfilename is None:
        modelfilename = default_model_filename(modelparams, birdparams)
    train_from_images(m, imagedir, p, modelfilename)


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
