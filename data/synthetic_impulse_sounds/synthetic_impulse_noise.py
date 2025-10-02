""" 
This script provides a set of classes and functions for generating synthetic impulse sounds with various characteristics. 
The script includes the following features:
1. **Truncated Normal Distribution Initialization**:
  - A utility function `initialize_distrib` to create truncated normal distributions for parameter sampling.
2. **Asymmetric Gaussian Envelope**:
  - A function `asymetric_gaussian_envelope` to generate an asymmetric Gaussian envelope for shaping signals.
3. **Modulated Chirp Signal**:
  - The `ModulatedChirp` class generates chirp signals with frequency modulation and an asymmetric Gaussian envelope.
4. **Modulated Harmonic Sum**:
  - The `ModulatedHarmonicSum` class generates signals composed of a sum of harmonics modulated by an asymmetric Gaussian envelope.
5. **Modulated White Noise**:
  - The `ModulatedWhiteNoise` class generates white noise signals filtered and modulated by an asymmetric Gaussian envelope.
6. **Autoregressive (AR) Impulse Noise**:
  - The `ARImpulseNoise` class generates impulse noise signals using autoregressive (AR) models with stable coefficients and modulated by an asymmetric Gaussian envelope.
"""

import numpy as np

from scipy.stats import truncnorm
import scipy.signal as sg

def initialize_distrib(mean, std, min_val, max_val):
  """
  Initialize a truncated normal distribution.

  Parameters
  ----------
  mean : float
    The mean (center) of the distribution.
  std : float
    The standard deviation (spread or width) of the distribution.
  min_val : float
    The minimum value of the distribution.
  max_val : float
    The maximum value of the distribution.

  Returns
  -------
  distribution : scipy.stats._distn_infrastructure.rv_frozen
    A frozen random variable object representing the truncated normal distribution.
  """
  a = (min_val - mean) / std
  b = (max_val - mean) / std

  distribution = truncnorm(
      a, b, loc=mean, scale=std)

  return distribution


def asymetric_gaussian_envelope(t, t_peak, tau_attack, tau_release):
  """
  Generate an asymmetric Gaussian envelope.
  Parameters
  ----------
  t : array_like
    Array of time values.
  t_peak : float
    The peak time where the envelope reaches its maximum value.
  tau_attack : float
    The time constant for the attack phase (time before the peak).
  tau_release : float
    The time constant for the release phase (time after the peak).
  Returns
  -------
  env : ndarray
    The asymmetric Gaussian envelope corresponding to the input time values.
  """
  env = np.zeros_like(t)
  attack_mask = t < t_peak
  release_mask = t >= t_peak
  
  sigma_attack = tau_attack / 3  
  sigma_release = tau_release / 3

  env[attack_mask] = np.exp(-((t[attack_mask] - t_peak)
                            ** 2) / (2 * sigma_attack**2))
  env[release_mask] = np.exp(-((t[release_mask] - t_peak)
                             ** 2) / (2 * sigma_release**2))

  return env

class ModulatedChirp():
  """
  A class to generate modulated chirp signals with configurable parameters.
  Parameters
  ----------
  sr : int
    The sampling rate of the signal in Hz.
  T_signal : float
    The total duration of the signal in seconds.
  params_config : dict or None
    A dictionary containing configuration parameters for the signal generation.
    If None, the `forward` method cannot be used. The dictionary should contain:
    - 'f0': dict
      Parameters for the fundamental frequency distribution.
    - 'temporal_support': dict
      Parameters for the temporal support distribution.
    - 'frequency_variation': dict
      A dictionary with a key 'values' containing a list of frequency variation values.
    - 'tau_attack': dict
      Parameters for the attack time proportion distribution.
      
  Methods
  -------
  generate(f0, k, tau_attack, temporal_support)
    Generate a modulated chirp signal with the specified parameters.
  forward()
    Generate a modulated chirp signal using the configured parameter distributions.
    
  Notes
  -----
  - The `generate` method creates a chirp signal with an asymmetric Gaussian envelope
    and adds a small amount of white noise.
  - The `forward` method samples parameters from the configured distributions to
    generate a signal.
  """
  
  def __init__(self, sr, T_signal, params_config):
    
    self.sr = sr
    self.t = np.arange(0, T_signal, 1/sr)
    
    self.params_config = params_config
    
    if self.params_config is not None:
      self.f0_params = params_config['f0']
      self.temporal_support_params = params_config['temporal_support']
      self.frequency_variation_params = params_config['frequency_variation']
      self.tau_attack_params = params_config['tau_attack']
      
      self.f0_distrib = initialize_distrib(**self.f0_params)
      self.temporal_support_distrib = initialize_distrib(**self.temporal_support_params)
      self.tau_attack_distrib = initialize_distrib(**self.tau_attack_params)
    
  def generate(self, f0, k, tau_attack, temporal_support):
    
    tau_release = temporal_support - tau_attack
    t_peak = tau_attack
    
    t_chirp = np.linspace(0, temporal_support, int(self.sr * temporal_support), endpoint=False)
    env = asymetric_gaussian_envelope(
      t_chirp, t_peak, tau_attack, tau_release
    )
    f = f0 + k * t_chirp
    chirp_signal = env*np.cos(2 * np.pi * (f*t_chirp))
    
    signal = np.zeros(len(self.t))
    signal[:len(chirp_signal)] = chirp_signal
    
    signal /= 1.1*np.max(np.abs(signal))
    
    white_noise = 0.0005*np.random.randn(len(signal))
  
    
    return signal + white_noise
    
    
  def forward(self):
    
    if self.params_config is None:
      raise ValueError("You must provide a params_config to generate the signal with the forward method.")
    
    f0 = self.f0_distrib.rvs()
    temporal_support = self.temporal_support_distrib.rvs()
    tau_attack_proportion = self.tau_attack_distrib.rvs()  # proportion of the temporal support
    tau_attack = tau_attack_proportion*temporal_support/2    
    
    k_ind = np.random.randint(0, len(self.frequency_variation_params['values']))
    k = self.frequency_variation_params['values'][k_ind]
    
    signal = self.generate(f0, k, tau_attack, temporal_support)
    
    return signal, temporal_support
  
  
class ModulatedHarmonicSum():
  """
  A class to generate modulated harmonic sum signals with an asymmetric Gaussian envelope.
  
  Parameters
  ----------
  sr : int
    Sampling rate of the signal.
  T_signal : float
    Duration of the signal in seconds.
  params_config : dict or None
    Configuration dictionary containing parameters for signal generation.
    Keys include:
    - 'f0': Parameters for the fundamental frequency distribution.
    - 'temporal_support': Parameters for the temporal support distribution.
    - 'tau_attack': Parameters for the attack time distribution.
    - 'n_harmonics': List of possible numbers of harmonics.
    
  Methods
  -------
  generate(f0, n_harmonics, tau_attack, temporal_support)
    Generate a single modulated harmonic sum signal.
  forward()
    Generate a signal using the configured parameter distributions.
  """
  
  
  def __init__(self, sr, T_signal, params_config):
    self.sr = sr
    self.t = np.arange(0, T_signal, 1/sr)
    
    self.params_config = params_config
    
    if self.params_config is not None:
      self.f0_params = params_config['f0']
      self.temporal_support_params = params_config['temporal_support']
      self.tau_attack_params = params_config['tau_attack']
      self.n_harmonics_params = params_config['n_harmonics']  # ou à fixer, à voir
      
      self.f0_distrib = initialize_distrib(**self.f0_params)
      self.temporal_support_distrib = initialize_distrib(**self.temporal_support_params)
      self.tau_attack_distrib = initialize_distrib(**self.tau_attack_params)
    
  def generate(self, f0, n_harmonics, tau_attack, temporal_support):
    
    tau_release = temporal_support - tau_attack
    t_peak = tau_attack
    
    t_impulse = np.linspace(0, temporal_support, int(self.sr * temporal_support), endpoint=False)
    env = asymetric_gaussian_envelope(
      t_impulse, t_peak, tau_attack, tau_release
    )    
    
    # Somme des harmoniques avec amplitudes décroissantes
    impulse = np.zeros_like(t_impulse)
    for n in range(1, n_harmonics + 1):
      impulse += (1 / n) * np.cos(2 * np.pi * n * f0 * t_impulse + np.random.randn())

    # Modulation par l'enveloppe gaussienne
    impulse *= env

    # Création du signal final
    signal = np.zeros_like(self.t)
    signal[:len(impulse)] = impulse    
    signal /= 1.1*np.max(np.abs(signal))    
    # White noise
    white_noise = 0.0005*np.random.randn(len(signal))
    
    return signal + white_noise
  
  def forward(self):
    
    if self.params_config is None:
      raise ValueError("You must provide a params_config to generate the signal with the forward method.")
    
    f0 = self.f0_distrib.rvs()
    n_harmonics_ind = np.random.randint(0, len(self.n_harmonics_params['values']))
    n_harmonics = self.n_harmonics_params['values'][n_harmonics_ind]
    temporal_support = self.temporal_support_distrib.rvs()
    tau_attack_proportion = self.tau_attack_distrib.rvs()  # proportion of the temporal support
    tau_attack = tau_attack_proportion*temporal_support/2   
    signal = self.generate(f0, n_harmonics, tau_attack, temporal_support)
    
    return signal, temporal_support
    
    
class ModulatedWhiteNoise():
  """
  A class to generate modulated white noise signals with configurable parameters.
  
  Parameters
  ----------
  sr : int
    Sampling rate of the signal in Hz.
  T_signal : float
    Duration of the signal in seconds.
  params_config : dict or None
    Configuration dictionary containing parameters for the signal generation.
    Keys include:
    - 'temporal_support': Parameters for the temporal support distribution.
    - 'tau_attack': Parameters for the attack time distribution.
    - 'fc': Parameters for the cutoff frequency distribution.
    If None, the `forward` method cannot be used.
    
  Methods
  -------
  generate(fc, tau_attack, temporal_support)
    Generate a single modulated white noise signal with specified parameters.
  forward()
    Generate a modulated white noise signal using parameters sampled from the configured distributions.
  """
  
  def __init__(self, sr, T_signal, params_config):
    self.sr = sr
    self.t = np.arange(0, T_signal, 1/sr)
    
    self.params_config = params_config
    
    if self.params_config is not None:
      self.temporal_support_params = params_config['temporal_support']
      self.tau_attack_params = params_config['tau_attack']
      self.fc_params = params_config['fc']

      self.temporal_support_distrib = initialize_distrib(**self.temporal_support_params)
      self.tau_attack_distrib = initialize_distrib(**self.tau_attack_params)
      self.fc_distrib = initialize_distrib(**self.fc_params)
    
  def generate(self, fc, tau_attack, temporal_support):
    
    tau_release = temporal_support - tau_attack
    t_peak = tau_attack
    
    t_impulse = np.linspace(0, temporal_support, int(self.sr * temporal_support), endpoint=False)
    env = asymetric_gaussian_envelope(
      t_impulse, t_peak, tau_attack, tau_release
    )    
    
    impulse = np.random.randn(len(t_impulse))
    
    impulse = sg.lfilter(*sg.butter(4, fc / (self.sr / 2)), impulse)
    
    impulse *= env

    signal = np.zeros_like(self.t)
    signal[:len(impulse)] = impulse    
    signal /= 1.1*np.max(np.abs(signal))    
    white_noise = 0.0005*np.random.randn(len(signal))
    
    return signal + white_noise
  
  def forward(self):
    
    if self.params_config is None:
      raise ValueError("You must provide a params_config to generate the signal with the forward method.")
    
    fc = self.fc_distrib.rvs()
    temporal_support = self.temporal_support_distrib.rvs()
    tau_attack_proportion = self.tau_attack_distrib.rvs()  # proportion of the temporal support
    tau_attack = tau_attack_proportion*temporal_support/2    
    signal = self.generate(fc, tau_attack, temporal_support)
    
    return signal, temporal_support
  
  
class ARImpulseNoise():
  """ 
  A class to generate synthetic impulse noise signals using an autoregressive (AR) model.
  
  Parameters
  ----------
  sr : int
    Sampling rate of the signal in Hz.
  T_signal : float
    Total duration of the signal in seconds.
  params_config : dict
    Configuration dictionary containing parameters for signal generation:
    - 'temporal_support': dict, parameters for the temporal support distribution.
    - 'f_min': float, minimum frequency for AR poles.
    - 'f_max': float, maximum frequency for AR poles.
    - 'radius_min': float, minimum radius for AR poles.
    - 'radius_max': float, maximum radius for AR poles.
    - 'tau_attack': dict, parameters for the attack time distribution.
    - 'ar_order': list of int, possible AR model orders.
    
  Methods
  -------
  random_stable_ar_coeffs(p, radius_max=0.99)
    Generate stable AR coefficients with real poles.
  random_stable_ar_coeffs_complex(p, fs=44100, f_min=50, f_max=16000, radius_max=0.99, radius_min=0.5)
    Generate stable AR coefficients with complex conjugate poles.
  generate(ar_coeffs, tau_attack, temporal_support)
    Generate a single impulse noise signal with a given AR model and envelope.
  forward()
    Generate a synthetic impulse noise signal using the configured parameters.
  """ 
  
  def __init__(self, sr, T_signal, params_config):
    self.sr = sr
    self.t = np.arange(0, T_signal, 1/sr)
    
    self.params_config = params_config
    
    if self.params_config is not None:
      self.temporal_support_params = params_config['temporal_support']
      self.f_min = params_config['f_min']
      self.f_max = params_config['f_max']
      self.radius_min = params_config['radius_min']
      self.radius_max = params_config['radius_max']
      self.tau_attack_params = params_config['tau_attack']
      self.ar_order_params = params_config['ar_order']

      self.temporal_support_distrib = initialize_distrib(**self.temporal_support_params)
      self.tau_attack_distrib = initialize_distrib(**self.tau_attack_params)

  def random_stable_ar_coeffs(self, p, radius_max=0.99):
    """
    Randomly generates AR coefficients ensuring stability.

    - p: order of the AR model
    - radius_max: maximum radius of the poles (must be < 1 for stability)

    Returns:
    - ar_coeffs: AR model coefficients ensuring a stable system
    """
    # Generate random poles inside the unit circle
    poles = np.random.uniform(-radius_max, radius_max, p)

    # Convert poles to AR coefficients via the characteristic polynomial
    ar_coeffs = np.poly(poles)[1:]  # Ignore the first coefficient (1)

    return -ar_coeffs  # Convention AR : 1 - Σ a_i z^-i


  def random_stable_ar_coeffs_complex(self, p, fs=44100, f_min=50, f_max=16000, radius_max=0.99, radius_min=0.5):
    """
    Generates AR coefficients with complex conjugate poles.

    - p: order of the AR model (must be even)
    - radius_max: maximum radius of the poles (close to 1 for long sustain)

    Returns:
    - ar_coeffs: AR model coefficients ensuring a stable system
    """
    assert p % 2 == 0, "The order of the AR model must be even to have complex conjugate poles."

    poles = []
    for _ in range(p // 2):
      r = np.random.uniform(radius_min, radius_max)  # Radius (damping)
      # Angle for the resonant frequency
      theta_min = 2 * np.pi * f_min / fs
      theta_max = 2 * np.pi * f_max / fs
      theta = np.random.uniform(theta_min, theta_max)      
      pole = r * np.exp(1j * theta)  # Complex pole
      poles.extend([pole, np.conj(pole)])  # Add the conjugate pole

    # Convert to AR coefficients
    ar_coeffs = np.poly(poles)[1:]  # Ignore the first coefficient (1)

    return -ar_coeffs.real  # Convention AR : 1 - Σ a_i z^-i
  
  def generate(self, ar_coeffs, tau_attack, temporal_support):
    
    tau_release = temporal_support - tau_attack
    t_peak = tau_attack
    
    t_impulse = np.linspace(0, temporal_support, int(self.sr * temporal_support), endpoint=False)
    env = asymetric_gaussian_envelope(
      t_impulse, t_peak, tau_attack, tau_release
    )   
    
    # Excitation 
    excitation = 0.5 * np.random.randn(len(t_impulse))
    
    # Apply AR filter
    ar_filter = [1] + [-a for a in ar_coeffs]  # AR process convention
    impulse = sg.lfilter([1], ar_filter, excitation)
    
    # Apply envelope
    impulse *= env
    
    signal = np.zeros_like(self.t)
    signal[:len(impulse)] = impulse       
    signal /= 1.1*np.max(np.abs(signal))   
     
    # White noise
    white_noise = 0.0005*np.random.randn(len(signal))
    
    signal = signal + white_noise
    
    return signal/np.max(np.abs(signal))
    
    
  def forward(self):
    
    if self.params_config is None:
      raise ValueError("You must provide a params_config to generate the signal with the forward method.")
    
    
    ar_order_ind = np.random.randint(0, len(self.ar_order_params))
    
    ar_coeffs = self.random_stable_ar_coeffs_complex(
      p=self.ar_order_params[ar_order_ind],
      radius_max=self.radius_max,
      radius_min=self.radius_min,
      f_min=self.f_min,
      f_max=self.f_max,
      fs=self.sr, 
      )
    temporal_support = self.temporal_support_distrib.rvs()
    tau_attack_proportion = self.tau_attack_distrib.rvs()  # proportion of the temporal support
    tau_attack = tau_attack_proportion*temporal_support/2
    
    
    signal = self.generate(ar_coeffs, tau_attack, temporal_support)
    
    return signal, temporal_support
    