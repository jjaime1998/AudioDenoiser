import tensorflow as tf
import os
from keras_radam import RAdam
import soundfile as sf
import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
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
from default_config import args
from data_processing.feature_extractor import FeatureExtractor
from tensorflow.python.client import device_lib
from utils import get_tf_feature, read_audio, write_audio, revert_features_to_audio, prepare_input_features, add_noise_to_clean_audio, tf_record_parser

os.environ['TF_KERAS'] = '1'
device_lib.list_local_devices()
#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

tf.random.set_seed(999)
np.random.seed(999)

def l2_norm(vector):
    return np.square(vector)

def SDR(denoised, cleaned, eps=1e-7): # Signal to Distortion Ratio
    a = l2_norm(denoised)
    b = l2_norm(denoised - cleaned)
    a_b = a / b
    return np.mean(10 * np.log10(a_b + eps))




if __name__ == '__main__':

   home_dir = args.home_dir
   mozilla_basepath = args.mozilla_basepath
   urbansound_basepath = args.urbansound_basepath

   windowLength = args.windowLength
   overlap      = args.overlap
   ffTLength    = args.ffTLength
   inputFs      = args.inputFs
   fs           = args.fs
   numFeatures  = ffTLength//2 + 1
   numSegments  = 8

   model = models.build_model(l2_strength=0.0)
   model.summary()

   model.load_weights(os.path.join(mozilla_basepath, 'denoiser_cnn_log_mel_generator.h5'))

   cleanAudio, sr = read_audio(os.path.join(mozilla_basepath, 'clips', 'common_voice_en_16526.mp3'), sample_rate=fs)
   print("Min:", np.min(cleanAudio),"Max:",np.max(cleanAudio))
 '''
   noiseAudio, sr = read_audio(os.path.join(urbansound_basepath, 'audio', 'fold10', '7913-3-0-0.wav'), sample_rate=fs)
   print("Min:", np.min(noiseAudio),"Max:",np.max(noiseAudio))
 '''
   cleanAudioFeatureExtractor = FeatureExtractor(cleanAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   stft_features = cleanAudioFeatureExtractor.get_stft_spectrogram()
   stft_features = np.abs(stft_features)
   print("Min:", np.min(stft_features),"Max:",np.max(stft_features))

   noisyAudio = add_noise_to_clean_audio(cleanAudio)
   noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

   def revert_features_to_audio2(features, magnitude, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
       if cleanMean and cleanStd:
          features = cleanStd * features + cleanMean

       phase = np.transpose(features, (1, 0))
       features = np.squeeze(magnitude)
       features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

       features = np.transpose(features, (1, 0))
       return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)
       #return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram_GL(np.abs(features))

   noisyPhase = np.angle(noise_stft_features)
   print(noisyPhase.shape)
   noise_stft_features = np.abs(noise_stft_features)
   noisyMagnitude = noise_stft_features

   mean = np.mean(noise_stft_features)
   std = np.std(noise_stft_features)
   noise_stft_features = (noise_stft_features - mean) / std


   predictors = prepare_input_features(noise_stft_features, numSegments, numFeatures)

   predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
   predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
   print('predictors.shape:', predictors.shape)

   STFTFullyConvolutional = model.predict(predictors)
   print(STFTFullyConvolutional.shape)

   denoisedAudioFullyConvolutional = revert_features_to_audio2(STFTFullyConvolutional, noisyMagnitude,  mean, std)
   print("Min:", np.min(denoisedAudioFullyConvolutional),"Max:",np.max(denoisedAudioFullyConvolutional))
 #  ipd.Audio(data=denoisedAudioFullyConvolutional, rate=fs) # load a local WAV file

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
   write_audio('./output/clean.wav', fs, cleanAudio)

   ax2.plot(noisyAudio)
   ax2.set_title("Noisy Audio")
   write_audio('./output/noisy.wav', fs, noisyAudio)

   ax3.plot(denoisedAudioFullyConvolutional)
   ax3.set_title("Denoised Audio")
   write_audio('./output/denoised.wav', fs, denoisedAudioFullyConvolutional)
   plt.show()


