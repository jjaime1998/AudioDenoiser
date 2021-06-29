from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from data_processing.urban_sound_8K import UrbanSound8K
from data_processing.dataset import Dataset
from default_config import args
import warnings

warnings.filterwarnings(action='ignore')

mcv = MozillaCommonVoiceDataset(args.mozilla_basepath, val_dataset_size=100)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

us8K = UrbanSound8K(args.urbansound_basepath, val_dataset_size=20)
noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()


config = {'windowLength': args.windowLength,
          'overlap': round(0.25 * args.windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=200)

train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=400)

## Create Test Set
clean_test_filenames = mcv.get_test_filenames()

noise_test_filenames = us8K.get_test_filenames()
noise_test_filenames = noise_test_filenames

test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
#test_dataset.create_tf_record(prefix='test', subset_size=1, parallel=False)
test_dataset.create_tf_record(prefix='test', subset_size=100, parallel=False)

