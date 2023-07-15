import os
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile as sf

input_dir = "TestingNQ"
output_dir_noise = "output_noise"
output_dir_timestretch = "output_timestretch"
output_dir_pitchshift = "output_pitchshift"
# output_dir_shift = "output_shift"

augment_noise = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)])
augment_timestretch = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)])
augment_pitchshift = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1.0)])
# augment_shift = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])

os.makedirs(output_dir_noise, exist_ok=True)
os.makedirs(output_dir_timestretch, exist_ok=True)
os.makedirs(output_dir_pitchshift, exist_ok=True)
# os.makedirs(output_dir_shift, exist_ok=True)

for filename in os.listdir(input_dir):
  if filename.endswith('.wav'):
    print(filename)
    
    audio, sr = librosa.load(os.path.join(input_dir, filename), sr=None)

    sf.write(os.path.join(output_dir_noise, filename), augment_noise(samples=audio, sample_rate=sr), sr)
    sf.write(os.path.join(output_dir_timestretch, filename), augment_timestretch(samples=audio, sample_rate=sr), sr)
    sf.write(os.path.join(output_dir_pitchshift, filename), augment_pitchshift(samples=audio, sample_rate=sr), sr)
    # sf.write(os.path.join(output_dir_shift, filename), augment_shift(samples=audio, sample_rate=sr), sr)