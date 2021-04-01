import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import os
os.environ['TF_KERAS'] = '1'
from keras_radam import RAdam

import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import IPython.display as ipd
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import models
from data_processing.feature_extractor import FeatureExtractor
# Load the TensorBoard notebook extension.

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

tf.random.set_seed(999)
np.random.seed(999)

def l2_norm(vector):
    return np.square(vector)

def SDR(denoised, cleaned, eps=1e-7): # Signal to Distortion Ratio
    a = l2_norm(denoised)
    b = l2_norm(denoised - cleaned)
    a_b = a / b
    return np.mean(10 * np.log10(a_b + eps))
def read_audio(filepath, sample_rate, normalize=True):
    # print(f"Reading: {filepath}").
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
      div_fac = 1 / np.max(np.abs(audio)) / 3.0
      audio = audio * div_fac
    return audio, sr
        
def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio

def play(audio, sample_rate):
    ipd.display(ipd.Audio(data=audio, rate=sample_rate))  # load a local WAV file


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

def prepare_input_features(stft_features):
    # Phase Aware Scaling: To avoid extreme differences (more than
    # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments

def revert_features_to_audio(features, phase, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    # features = librosa.db_to_power(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)

if __name__ == '__main__':

   home_dir = '/home/csprh/MYCODE/AUDIO/cnn_denoiser/'

   train_tfrecords_filenames = glob.glob(os.path.join(home_dir, 'records/train_*'))
   np.random.shuffle(train_tfrecords_filenames)
   train_tfrecords_filenames = list(train_tfrecords_filenames)
   print(train_tfrecords_filenames)
   val_tfrecords_filenames =  glob.glob(os.path.join(home_dir, 'records/val_*'))

   windowLength = 256
   overlap      = round(0.25 * windowLength) # overlap of 75%
   ffTLength    = windowLength
   inputFs      = 48e3
   fs           = 16e3
   numFeatures  = ffTLength//2 + 1
   numSegments  = 8
   print("windowLength:",windowLength)
   print("overlap:",overlap)
   print("ffTLength:",ffTLength)
   print("inputFs:",inputFs)
   print("fs:",fs)
   print("numFeatures:",numFeatures)
   print("numSegments:",numSegments)

   mozilla_basepath = '/media/csprh/B0C87E48C87E0CBA/Data/'
   urbansound_basepath = '/media/csprh/B0C87E48C87E0CBA/Data/UrbanSound8K'


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

   tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)


   #model.load_weights('drive/My Drive/datasets/dataset_v2/denoiser_cnn_log_mel_generator.h5')
   #baseline_val_loss = model.evaluate(test_dataset)[0]
   #print(baseline_val_loss)


   #early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, baseline=baseline_val_loss)
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

   # val_loss = model.evaluate(test_dataset)[0]
   # if val_loss < baseline_val_loss:
   #   print("New .model saved")
   #   model.save('drive/My Drive/datasets/dataset_v2/denoiser_cnn_log_mel_generator.h5')

   # model.save('drive/My Drive/datasets/dataset_v2/denoiser_cnn_generator.h5')
   # model.load_weights('drive/My Drive/datasets/one_noise_data/model.h5')
   model.load_weights(os.path.join(mozilla_basepath, 'denoiser_cnn_log_mel_generator.h5'))

   cleanAudio, sr = read_audio(os.path.join(mozilla_basepath, 'clips', 'common_voice_en_16526.mp3'), sample_rate=fs)
   print("Min:", np.min(cleanAudio),"Max:",np.max(cleanAudio))
   ipd.Audio(data=cleanAudio, rate=sr) # load a local WAV file

   noiseAudio, sr = read_audio(os.path.join(urbansound_basepath, 'audio', 'fold10', '7913-3-0-0.wav'), sample_rate=fs)
   print("Min:", np.min(noiseAudio),"Max:",np.max(noiseAudio))
   ipd.Audio(data=noiseAudio, rate=sr) # load a local WAV file

   cleanAudioFeatureExtractor = FeatureExtractor(cleanAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   stft_features = cleanAudioFeatureExtractor.get_stft_spectrogram()
   stft_features = np.abs(stft_features)
   print("Min:", np.min(stft_features),"Max:",np.max(stft_features))

   noisyAudio = add_noise_to_clean_audio(cleanAudio, noiseAudio)
   ipd.Audio(data=noisyAudio, rate=fs) # load a local WAV file


   noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

   # Paper: Besides, spectral phase was not used in the training phase.
   # At reconstruction, noisy spectral phase was used instead to
   # perform in- verse STFT and recover human speech.
   noisyPhase = np.angle(noise_stft_features)
   print(noisyPhase.shape)
   noise_stft_features = np.abs(noise_stft_features)

   mean = np.mean(noise_stft_features)
   std = np.std(noise_stft_features)
   noise_stft_features = (noise_stft_features - mean) / std

   predictors = prepare_input_features(noise_stft_features)

   predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
   predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
   print('predictors.shape:', predictors.shape)

   STFTFullyConvolutional = model.predict(predictors)
   print(STFTFullyConvolutional.shape)


   denoisedAudioFullyConvolutional = revert_features_to_audio(STFTFullyConvolutional, noisyPhase, mean, std)
   print("Min:", np.min(denoisedAudioFullyConvolutional),"Max:",np.max(denoisedAudioFullyConvolutional))
   ipd.Audio(data=denoisedAudioFullyConvolutional, rate=fs) # load a local WAV file

   # A numeric identifier of the sound class -- Types of noise
   # 0 = air_conditioner
   # 1 = car_horn
   # 2 = children_playing
   # 3 = dog_bark
   # 4 = drilling
   # 5 = engine_idling
   # 6 = gun_shot
   # 7 = jackhammer
   # 8 = siren
   # 9 = street_music

   f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

   ax1.plot(cleanAudio)
   ax1.set_title("Clean Audio")

   ax2.plot(noisyAudio)
   ax2.set_title("Noisy Audio")

   ax3.plot(denoisedAudioFullyConvolutional)
   ax3.set_title("Denoised Audio")


