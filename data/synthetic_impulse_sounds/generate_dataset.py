""" 
Generation of synthetic impulse sound dataset.
"""

import numpy as np 
import librosa 
import os 
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from config import generation_parameters

from synthetic_impulse_noise import ModulatedChirp, ModulatedHarmonicSum, ARImpulseNoise

parent_data_folder = generation_parameters["saving"]["parent_data_folder"]
saving_audio_path = generation_parameters["saving"]["audio_path"]

# Create folders if needed
for folder in ["chirps", "harmonics", "ar_impulses"]:
  folder_path = os.path.join(parent_data_folder, saving_audio_path, "audio", folder)
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

sr = generation_parameters["sr"]
duration = generation_parameters["duration"]
N_impulses = generation_parameters["N_impulses"]
N_harmonics = generation_parameters["N_harmonics"]
N_chirps = generation_parameters["N_chirps"]
N_ar_impulses = generation_parameters["N_ar_impulses"]

# Generators

chirp_generator = ModulatedChirp(
  sr=sr,
  T_signal=duration,
  params_config=generation_parameters["type_parameters"]["chirp"]
)

harmonic_generator = ModulatedHarmonicSum(
  sr=sr,
  T_signal=duration,
  params_config=generation_parameters["type_parameters"]["harmonic"]
)

ar_impulse_generator = ARImpulseNoise(
  sr=sr,
  T_signal=duration,
  params_config=generation_parameters["type_parameters"]["ar_impulse"]
)

# Metadata initialization
label = ['chirp']*N_chirps + ['harmonic']*N_harmonics + ['ar_impulse']*N_ar_impulses
duration = [duration]*N_impulses
temporal_supports = []
file_paths = []

# Chirps
for i in tqdm(range(N_chirps)):
  saving_path = os.path.join(
    "audio", 
    "chirps",
    f"chirp_{i}.wav"
  )
  file_path = os.path.join(parent_data_folder,
                           saving_audio_path,
                           saving_path)
  file_paths.append(saving_path)
  audio, temporal_support = chirp_generator.forward()
  temporal_supports.append(temporal_support)  
  sf.write(
    file=file_path, 
    data=audio, 
    samplerate=sr
  )
  
# Harmonics
for i in tqdm(range(N_harmonics)):
  saving_path = os.path.join(
    "audio",
    "harmonics",
    f"harmonic_{i}.wav"
  )
  file_path = os.path.join(parent_data_folder,
                           saving_audio_path,
                           saving_path)
  file_paths.append(saving_path)
  audio, temporal_support = harmonic_generator.forward()
  temporal_supports.append(temporal_support)
  sf.write(
    file=file_path, 
    data=audio, 
    samplerate=sr
  )
  
# AR Impulses
for i in tqdm(range(N_ar_impulses)):
  saving_path = os.path.join(
    audio,
    "ar_impulses",
    f"ar_impulse_{i}.wav"
  )
  file_path = os.path.join(parent_data_folder,
                           saving_audio_path,
                           saving_path)
  file_paths.append(saving_path)
  audio, temporal_support = ar_impulse_generator.forward()
  temporal_supports.append(temporal_support)
  sf.write(
    file=file_path, 
    data=audio, 
    samplerate=sr
  )
  
  
# Saving metadata in a dataframe
metadata = {
  "label": label,
  "duration": duration,
  "temporal_support": temporal_supports,
  "file_path": file_paths
}

metadata_path = os.path.join(
  parent_data_folder,
  generation_parameters["saving"]["metadata_path"], 
  "metadata.csv"
)

dataframe = pd.DataFrame(metadata)
dataframe.to_csv(metadata_path, index=False)