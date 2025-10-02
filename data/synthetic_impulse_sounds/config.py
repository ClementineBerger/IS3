import numpy as np
import os

# General parameters

N_impulses = 3000
N_harmonics = int(0.3*N_impulses)
N_chirps = int(0.3*N_impulses)
N_ar_impulses = N_impulses - N_harmonics - N_chirps

parent_data_folder = os.environ["DATA_DIR"]

# or your own chosen path
saving_audio_path = "synthetic_impulse_sounds"

saving_metadata_path = "synthetic_impulse_sounds"

chirp = {
  "temporal_support":{"mean": 0.1, "std": 0.1, "min_val": 0.02, "max_val": 0.3},
  "f0": {"mean": 500, "std": 600, "min_val": 150, "max_val": 2000},
  "tau_attack":{ "mean": 0.2, "std": 0.2, "min_val": 0.01, "max_val": 0.8}, # proportion of temporal_support / 2
  "frequency_variation": {"values": [1, 10, 100, 1000]},
}

harmonic = {
  "n_harmonics": {"values": [2, 5, 8, 10, 50, 100]},
  "f0": {"mean": 500, "std": 600, "min_val": 40, "max_val": 2000},
  "temporal_support":{"mean": 0.1, "std": 0.2, "min_val": 0.02, "max_val": 0.4},
  "tau_attack":{ "mean": 0.2, "std": 0.2, "min_val": 0.01, "max_val": 0.8}, # proportion of temporal_support / 2
}

ar_impulse = {
  "f_min": 40,
  "f_max": 16000,
  "ar_order": [2, 4, 6, 8, 10],
  "radius_min": 0.5, 
  "radius_max": 0.99,
  "temporal_support":{"mean": 0.1, "std": 0.2, "min_val": 0.02, "max_val": 0.4},
  "tau_attack":{ "mean": 0.2, "std": 0.2, "min_val": 0.01, "max_val": 0.8}, # proportion of temporal_support / 2  
}

generation_parameters = {
  "sr": 44100,
  "duration": 0.5,
  "N_impulses": N_impulses,
  "N_harmonics": N_harmonics,
  "N_chirps": N_chirps,
  "N_ar_impulses": N_ar_impulses,
  "type_parameters": {"chirp": chirp, "harmonic": harmonic, "ar_impulse": ar_impulse},
  "saving":{
    "parent_data_folder": parent_data_folder,
    "audio_path": saving_audio_path,
    "metadata_path": saving_metadata_path
  },
}