# Deepchirp
(a zebra finch syllable automatic labeling system)

Deepchirp uses deep learning to automate identification and labelling in zebra finch microphone recordings. It has been designed to be robust to noise artifacts such as conspecific calls and movement artifacts.

Deepchirp uses Keras, a wrapper around TensorFlow, to efficiently train deep learning models on GPUs.

Deepchirp was inspired by [Koumura & Okanoya 2016](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159188).

# installation

Requires Python 3+

Required Python libaries:
+ keras >= 2.0
+ [bark](https://github.com/kylerbrown/bark)
+ [resin](https://github.com/kylerbrown/resin)


# Usage
## Step 1. Manually label data
Use either Praat or bark's `dat-segment` and `bark-label-view` to create a training set by manually labeling syllables.

## Step 1.5 (optional) enrich and augment training data.

Most continuous recordings are filled mostly with silence. The program `dat-enrich` from the Bark library helps remove an excessive amout of silence. Before running `dat-enrich` you may want to use `enrich_csv.py`,
which cleans up labels and adds new labels for silence near syllables so they can be more heavily sampled.

## Step 2. create training and test datasets

Use `to_images.py` to create training and test samples. 

Example usage:

`python to_images.py path/to/enriched/training/data mymodel.yaml mybird.yaml`

Get help with `python to_images.py -h`.

This script creates nested folders like this:
```
model_bird/
    train/
        i/
        a/
        b/
    test/
        i/
        a/
        b/
        (etc)
        etc
```
Each subdirectory contains examples of the syllable with the directory's name. For example, `i/` contains intro notes.

The script `to_images.py` requires two parameters files. These files must use YAML syntax. The first parameters file determines how images are created from the signal using spectrograms, and which model to train on. The following parameters are required:
+ sr -- sampling rate
+ NFFT -- number of points in the Fourier transform
+ data\_window -- number data points to perform the FFT on, must be less than or equal to NFFT
+ freq\_range -- A minimum and maximum frequence to include
+ n\_tapers -- number of tapers to use in the creation of multitaper spectrograms
+ window\_spacing -- distance between two FFT windows in points
+ window\_len -- the number of sequenctial FFTs to include on either side of the target moment in time. The total size of a spectram for training is `2 * window_len + 1`.
+ NW -- a bandwidth parameter for the multitaper spectrogram.
+ model -- the name of the model to use. Must match the name of a function defined in `model.py`
+ batch\_size -- number of training examples to use to estimate the gradient and update the model weights.
+ predict\_batch\_size -- number of predictions to generate in a single batch, tune for speed, does not effect accuracy.
+ steps\_per\_epoch -- Number of batches to lump together for training
+ epochs -- total number of epochs. The number of training samples used is `batch_size * steps_per_epoch * epochs`.
+ amplitude\_norm -- a constant to scale the spectrogram values. Possibly useful to restrict the input values to the range (0, 1).
+ max\_samples\_per\_category -- maximum number of training samples to create for each category.


An example parameters file:

```yaml
# spectral parameters
sr: 30000
NFFT: 256
data_window: 256
freq_range:
    - 300
    - 8000
n_tapers: 2
window_spacing: 200
window_len: 40
NW: 1.5

# model parameters
model: simplecnn3
batch_size: 32
predict_batch_size: 256
steps_per_epoch: 500
epochs: 25
amplitude_norm: 100000
max_samples_per_category: 40000
```


A second parameters file is also required, which contains bird specific parameters -- the target categories and any other categories that should be trained to be ignored. Target `0` is reserved for silence.

Example:
```yaml
# target parameters
encoder:
    i: 1
    a: 2
    b: 3
    c: 4
    d: 5

# used only to draw even pools of samples
# during training, they are classified as silence
sampling_encoder_extras:
    - y
    - z
```

In the above example, `y` and `z` samples will be saved, but their target category will be `0` (silence). This is useful for training the model to ignore cage noise, calls, or to pay extra attention to the silences between syllables to improve onset and offset detection.

Example parameters files are in the `examples/` folder. You can use these as templates to create your own.

## Step 3. Train
Using the training images created by `to_images.py`, train the network. Example usage:

    python train.py mymodel.yaml mybird.yaml

## Step 4. Test

Test the perfomance of model on withheld samples:

    python test.py mymodel.yaml mybird.yaml

## Step 5. Predict
Apply the model to new raw data. Data must be in raw binary form with valid Bark metadata files

    predict.py mymodel.yaml mybird.yaml path/to/mybird/recordings

