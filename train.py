import tensorflow as tf
import os
from keras_radam import RAdam
import librosa
import pandas as pd
import os
import datetime
import numpy as np
import tensorflow as tf
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import models
from tensorflow.python.client import device_lib
from default_config import args

os.environ['TF_KERAS'] = '1'
device_lib.list_local_devices()
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


tf.random.set_seed(999)
np.random.seed(999)


def tf_record_parser(record):
    keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
    clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
    noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

    # reshape input and annotation images
    noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (129, 8, 1), name="noise_stft_mag_features")
    clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (129, 1, 1), name="clean_stft_magnitude")
    noise_stft_phase = tf.reshape(noise_stft_phase, (129,), name="noise_stft_phase")

    return noise_stft_mag_features, clean_stft_magnitude


if __name__ == '__main__':

   home_dir = args.home_dir
   mozilla_basepath = args.mozilla_basepath
   urbansound_basepath = args.urbansound_basepath

   train_tfrecords_filenames = glob.glob(os.path.join(home_dir, 'records/train_*'))
   np.random.shuffle(train_tfrecords_filenames)
   train_tfrecords_filenames = list(train_tfrecords_filenames)
   print(train_tfrecords_filenames)
   val_tfrecords_filenames =  glob.glob(os.path.join(home_dir, 'records/val_*'))

   windowLength = args.windowLength
   overlap      = args.overlap
   ffTLength    = args.ffTLength
   inputFs      = args.inputFs
   fs           = args.fs

   numFeatures  = ffTLength//2 + 1
   numSegments  = 8
   print("windowLength:",windowLength)
   print("overlap:",overlap)
   print("ffTLength:",ffTLength)
   print("inputFs:",inputFs)
   print("fs:",fs)
   print("numFeatures:",numFeatures)
   print("numSegments:",numSegments)

   train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
   train_dataset = train_dataset.map(tf_record_parser)
   train_dataset = train_dataset.shuffle(8192)
   train_dataset = train_dataset.repeat()
   train_dataset = train_dataset.batch(512+256)
   train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

   test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
   test_dataset = test_dataset.map(tf_record_parser)
   test_dataset = test_dataset.repeat(1)
   test_dataset = test_dataset.batch(512)


   model = models.build_model(l2_strength=0.0)
   model.summary()

#   tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

   early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

   logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
   tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
   checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(mozilla_basepath, 'denoiser_cnn_log_mel_generator.h5'), 
                                                         monitor='val_loss', save_best_only=True)

   model.fit(train_dataset,
         steps_per_epoch=600,
         validation_data=test_dataset,
         epochs=9999,
         callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
        )


