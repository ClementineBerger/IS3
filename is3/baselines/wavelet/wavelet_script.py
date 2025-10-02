import numpy as np
import matplotlib.pyplot as plt

import librosa

import pywt

import scipy.signal as sig
from scipy.ndimage import median_filter

sr = 44100

class WaveletBaseline():
  """
  A class for wavelet-based signal processing, including impulse detection 
  and signal reconstruction.
  
  Based on R. C. Nongpiur. Impulse noise removal in speech using wavelets. In ICASSP, 2008.
  
  Parameters
  ----------
  wavelet : str
    The name of the wavelet to use.
  level : int
    The decomposition level for the wavelet transform.
  sr : int
    The sampling rate of the input signal.
  ks : float
    Scaling factor for the dynamic threshold.
  ks_impulse : float or None
    Scaling factor for impulse detection threshold. If None, the same 
    threshold as `ks` is used.
  kc : float
    Scaling factor for coarse-scale impulse detection.
  kernel_size : int, optional
    The maximum kernel size for median filtering, by default 1025.
  Methods
  -------
  compute_coeffs(x)
    Computes the wavelet decomposition coefficients of the input signal.
  compute_tau(coeffs, kernel_size)
    Computes the dynamic threshold for a given set of coefficients.
  detect_impulses_on_fine_scale(coeffs, kernel_size)
    Detects impulses on the fine-scale coefficients using a dynamic threshold.
  detect_impulses_on_coarse_scale(coeffs_coarse, coeffs_finest)
    Detects impulses on the coarse-scale coefficients using the finest-scale 
    coefficients for reference.
  update_coefficients(coeffs)
    Updates the wavelet coefficients by separating background and impulse 
    components.
  reconstruct_signal(coeffs)
    Reconstructs the signal from the given wavelet coefficients.
  forward(x)
    Processes the input signal to separate background and impulse components 
    and reconstructs the corresponding signals.
  """
  def __init__(self, wavelet, level, sr, ks, ks_impulse, kc, kernel_size=1025):
    self.wavelet = wavelet + str(level)
    self.level = level
    self.sr = sr
    self.ks = ks
    self.ks_impulse = ks_impulse
    self.kc = kc
    self.kernel_size = kernel_size
    
  def compute_coeffs(self, x):
    coeffs = pywt.wavedec(x, self.wavelet, level=self.level)
    return coeffs
  
  def compute_tau(self, coeffs, kernel_size):
    """ Calcule le seuil dynamique """
    median_filtered = median_filter(
      np.abs(coeffs),
      size=kernel_size, mode='reflect')
    threshold = self.ks * median_filtered
    return threshold
  
  def detect_impulses_on_fine_scale(
    self,
    coeffs,
    kernel_size):
    """ Détecte les impulsions en fonction du seuil dynamique """

    threshold = self.compute_tau(coeffs,kernel_size)
    
    if self.ks_impulse is not None :
      thresh_impulse = self.ks_impulse*threshold/self.ks
    else: 
      thresh_impulse = threshold

    coeffs_background = np.where(
        np.abs(coeffs) > threshold,
        threshold *
        np.sign(coeffs),
        coeffs)

    coeffs_impulse = np.where(
        np.abs(coeffs) < thresh_impulse,
        0.,
        coeffs)

    return coeffs_background, coeffs_impulse  
  
  def detect_impulses_on_coarse_scale(
    self,
    coeffs_coarse,
    coeffs_finest,):
    tau_sf = self.compute_tau(coeffs_finest, kernel_size=self.kernel_size)
    val = np.abs(coeffs_finest) - tau_sf

    k = len(coeffs_finest) / len(coeffs_coarse)
    indices = np.round(np.arange(len(coeffs_coarse)) * k).astype(int)

    K = coeffs_coarse - self.kc * val[indices]

    coeffs_background = np.where(K > 0, K, 0)
    coeffs_impulse = 1e-12 * coeffs_coarse   # ?

    return coeffs_background, coeffs_impulse  
  
  def update_coefficients(self, coeffs):
    
    filtered_coeffs_background = []
    filtered_coeffs_impulse = []    
    
  
    for i, c in enumerate(coeffs):
      max_kernel_size = self.kernel_size
      if i <= len(coeffs) // 4:
        value = max_kernel_size // 4
      elif i <= len(coeffs) // 2:
        value = max_kernel_size // 2
      else:
        value = max_kernel_size
      kernel_size = value if value % 2 == 1 else value + 1
      if i <= len(coeffs) // 4:
        coeffs_background, coeffs_impulse = self.detect_impulses_on_coarse_scale(
            coeffs_coarse=c,
            coeffs_finest=coeffs[-1],)
        filtered_coeffs_background.append(coeffs_background)
        filtered_coeffs_impulse.append(coeffs_impulse)
      else:
        coeffs_background, coeffs_impulse = self.detect_impulses_on_fine_scale(
            c, kernel_size=min(len(c), kernel_size))
        filtered_coeffs_background.append(coeffs_background)
        filtered_coeffs_impulse.append(coeffs_impulse)
        
    return filtered_coeffs_background, filtered_coeffs_impulse
  
  def reconstruct_signal(self, coeffs):
    return pywt.waverec(coeffs, self.wavelet)
  
  def forward(self, x):
    coeffs = self.compute_coeffs(x)
    filtered_coeffs_background, filtered_coeffs_impulse = self.update_coefficients(coeffs)
    return self.reconstruct_signal(filtered_coeffs_background), self.reconstruct_signal(filtered_coeffs_impulse)