""" 
Synthetic Stationary Background generation
"""

import numpy as np

from audiomentations import Compose, AddGaussianSNR, LowPassFilter, GainTransition, SevenBandParametricEQ, ApplyImpulseResponse


metadata_irs = None

if metadata_irs is None:
  raise ValueError("Please set the metadata_irs variable to the path of the metadata CSV file for the impulse responses.")

filtered_metadata = metadata_irs[(metadata_irs['ir_type'].isin(
    ['rir', 'ir'])) & (metadata_irs['rt60_median'] <= 0.3)]
ir_paths = list(filtered_metadata['file_path'].values)


def generate_noise(noise_type, duration, fs):
  """Generates white, pink, or brown noise."""
  samples = int(duration * fs)
  if noise_type == "white":
    return np.random.normal(0, 1, samples)
  elif noise_type == "pink":
    # Pink noise (1/f noise): via inverse FFT
    uneven = samples % 2
    X = np.random.randn(
        samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    # Amplitude inversely proportional to frequency
    S = np.sqrt(np.arange(len(X)) + 1.)
    y = (np.fft.irfft(X / S)).real
    return y[:samples]
  elif noise_type == "brown":
    # Brown noise (cumulative)
    white_noise = np.random.normal(0, 1, samples)
    return np.cumsum(white_noise) / fs
  else:
    raise ValueError("Unsupported noise type")
  
  
class BackgroundSyntheticNoise():
  def __init__(self, sr, T_signal, params_config):
    
    self.sr = sr
    self.params_config = params_config
    self.duration = T_signal
    
    if self.params_config is not None:
      
      self.base_noise_type = params_config["base_noise_type"]
      
      aug = self.params_config["augmentations"]
      
      self.augmentations = Compose([
        AddGaussianSNR(**aug["gaussian_snr"]),
        LowPassFilter(**aug["low_pass_filter"]),
        SevenBandParametricEQ(**aug["seven_band_parametric_eq"]),
        GainTransition(**aug["gain_transition"]),
        ApplyImpulseResponse(ir_path=ir_paths, **aug["apply_impulse_response"]),
      ])
      
  def forward(self):
    
    base_noise = generate_noise(
      noise_type=self.base_noise_type,
      duration=self.duration,
      fs=self.sr
    )
    
    augmented_noise = self.augmentations(samples=base_noise, sample_rate=self.sr)
    augmented_noise -= np.mean(augmented_noise)
    
    return augmented_noise/np.max(np.abs(augmented_noise))