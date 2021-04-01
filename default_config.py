
class args(object):

   home_dir            = '/home/csprh/MYCODE/AUDIO/cnn_denoiser/' 
   mozilla_basepath    = '/media/csprh/B0C87E48C87E0CBA/Data/'
   urbansound_basepath = '/media/csprh/B0C87E48C87E0CBA/Data/UrbanSound8K'
   windowLength        = 256
   ffTLength           = windowLength
   inputFs             = 48e3
   fs                  = 16e3
   overlap             = round(0.25 * windowLength) # overlap of 75%
   numSegments         = 8
