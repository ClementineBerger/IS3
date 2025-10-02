"""
This file contains the configuration for the synthetic 
background data generation.
"""

import os

N_signals = 1500

parent_data_folder = os.environ["DATA_DIR"]

# or your own chosen path
saving_audio_path = "synthetic_impulse_sounds"

saving_metadata_path = "synthetic_impulse_sounds"

generation_parameters = {
  "sr": 44100,
  "duration": 10.,
  "N_signals": N_signals,
  "base_noise_type": "pink",
  "augmentations":{
    "gaussian_snr": {"min_snr_db": -3, "max_snr_db": 15, "p": 1.},
    "low_pass_filter": {"min_cutoff_freq": 200, "max_cutoff_freq": 2000, "p": 1.},
    "gain_transition": {"min_gain_db": -10, "max_gain_db": 10, "min_duration": 1., "max_duration": 10.,"p": 1.},
    "seven_band_parametric_eq": {"p": .5},
    "apply_impulse_response": {"p": 1.},
    "max_rt_60": 0.3
  },
  "saving":{
    "parent_data_folder": parent_data_folder,
    "audio_path": saving_audio_path,
    "metadata_path": saving_metadata_path
  },
}