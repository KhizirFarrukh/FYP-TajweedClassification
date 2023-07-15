import os

import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt

def create_spectrogram(audio_file, image_file):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

  y, sr = librosa.load(audio_file)
  ms = librosa.feature.melspectrogram(y=y, sr=sr)
  log_ms = librosa.power_to_db(ms, ref=np.max)
  librosa.display.specshow(log_ms, sr=sr)

  fig.savefig(image_file)
  plt.close(fig)

output_folder = os.path.join(os.getcwd(), 'spectrograms')

if not os.path.exists(output_folder):
  os.makedirs(output_folder)

# current_directory = os.getcwd()
# parent_directory = os.path.dirname(current_directory)
# input_folder = os.path.join(parent_directory, "test_dataset", "qalqalah (wav files)")
input_folder = "TestingNQ"

audio_files = os.listdir(input_folder)

for filename in audio_files:
  if filename.endswith('.wav'):
    print(filename)
    
    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename.replace('.wav', '.png'))

    create_spectrogram(input_file, output_file)