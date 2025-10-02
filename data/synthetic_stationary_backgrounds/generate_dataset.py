""" 
Generation of synthetic stationary background dataset.
"""

import numpy as np 
import librosa 
import os 
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from config import generation_parameters

from synthetic_stationary_background import BackgroundSyntheticNoise

parent_data_folder = generation_parameters["saving"]["parent_data_folder"]
saving_audio_path = generation_parameters["saving"]["audio_path"]

# Create folders if needed
if not os.path.exists(os.path.join(parent_data_folder, saving_audio_path, "audio")):
  os.makedirs(os.path.join(parent_data_folder, saving_audio_path, "audio"))
  
sr = generation_parameters["sr"]
duration = generation_parameters["duration"]
N_signals = generation_parameters["N_signals"]

# Generator
background_generator = BackgroundSyntheticNoise(
  sr=sr,
  T_signal=duration,
  params_config=generation_parameters
)

# Metadata initialization
label = ['synthetic_stationary']*N_signals
duration = [duration]*N_signals
file_paths = []

# Background
for i in tqdm(range(N_signals)):
  saving_path = os.path.join(
    "audio", 
    f"background_{i}.wav"
  )
  file_paths.append(saving_path)
  file_path=os.path.join(parent_data_folder, 
                         saving_audio_path,
                         saving_path)
  signal = background_generator.forward()
  # signal = signal.astype(np.float32)
  sf.write(file_path, signal, sr)
  
metadata = {
  "label": label,
  "duration": duration,
  "file_path": file_paths
}

metadata_path = os.path.join(
  parent_data_folder,
  generation_parameters["saving"]["metadata_path"],
  "metadata.csv"
)

dataframe = pd.DataFrame(metadata)
dataframe.to_csv(metadata_path, index=False)