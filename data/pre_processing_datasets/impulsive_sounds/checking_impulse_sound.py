"""
Code to check for a given audio file if it contains only on or a few impulsive sounds (with no background noise).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import pandas as pd

from numpy.lib import stride_tricks

import librosa

def compute_energy(signal, sr, rms):

  # If stereo, convert to mono
  if len(signal.shape) == 2 and signal.shape[0] > 1:
    signal = signal.mean(axis=1)

  # signal = signal.reshape(-1, 1)
  signal = signal.flatten()

  if rms:
    # Parameters of the sliding window
    window_size = int(0.02 * sr) 
    hop_size = int(0.01 * sr) 

    # Use sliding_window_view to obtain sliding windows
    windows = stride_tricks.sliding_window_view(
      signal, window_shape=window_size)[::hop_size]

    # Calculate the RMS level for each window
    rms_values = np.sqrt(np.mean(windows**2, axis=1))

    return rms_values

  else:
    return signal**2


def robust_max(energy, energy_threshold):

  robust_max = np.quantile(energy, energy_threshold)

  return robust_max


def compute_silence_proportion(
        signal,
        sr,
        rms,
        energy_threshold,
        min_threshold,
        plot=False):
  """
  Compute the proportion of silence in an audio signal based on energy levels.

  Parameters
  ----------
  signal : numpy.ndarray
    The input audio signal as a 1D array.
  sr : int
    The sampling rate of the audio signal in Hz.
  rms : float
    The root mean square (RMS) value of the signal.
  energy_threshold : float
    The threshold used to compute the robust maximum energy.
  min_threshold : float
    The proportion of the maximum energy used to define the silence threshold.
  plot : bool, optional
    If True, plots the energy levels and thresholds for visualization. 
    Default is False.

  Returns
  -------
  float
    The proportion of the signal that is considered silent, 
    where silence is defined as energy below the silence threshold.
  """

  energy = compute_energy(signal=signal, sr=sr, rms=rms)

  maximum = robust_max(energy, energy_threshold)

  if maximum < 1e-8:
    return 1.  # complete silence

  minimum = maximum * min_threshold

  if plot:
    plt.figure()
    plt.plot(energy)
    plt.axhline(y=maximum, color='red', label='Maximum energy')
    plt.axhline(y=minimum, color='green', label="Silence threshold")
    plt.legend(loc='upper right')
    plt.show()

  return np.mean(energy < minimum)


def onset_detection(signal, sr, delta, plot_figure=False):
  """ 
  Detects onset events in an audio signal.
  
  Parameters
  ----------
  signal : numpy.ndarray
    The input audio signal.
  sr : int
    Sampling rate of the audio signal.
  delta : float
    Threshold for onset detection.
  plot_figure : bool, optional
    If True, plots the signal, RMS values, and detected onsets (default is False).
    
  Returns
  -------
  numpy.ndarray
    Array of onset sample indices.
  """
  
  onset_hop_size = int(0.01 * sr)  # 25 ms
  
  window_size = int(0.02 * sr) 
  hop_size = int(0.01 * sr) 
  windows = stride_tricks.sliding_window_view(
      signal, window_shape=window_size)[::hop_size]

  rms_values = np.sqrt(np.mean(windows**2, axis=1))  
  
  onset_envelope = librosa.onset.onset_strength(y=signal, sr=sr) #, hop_length=onset_hop_size)
  onsets = librosa.onset.onset_detect(
      y=signal,
      sr=sr,
      hop_length=onset_hop_size,
      units='samples',
      normalize=True,
      #onset_envelope=onset_envelope,
      delta=delta)
  
  if plot_figure:
    
    window_size = int(0.1 * sr) 
    hop_size = int(0.01 * sr) 

    windows = stride_tricks.sliding_window_view(
        signal, window_shape=window_size)[::hop_size]

    rms_values = np.sqrt(np.mean(windows**2, axis=1))     
    
    plt.figure()
    t = np.arange(len(signal))/sr
    t_hop_size = np.arange(len(rms_values))*hop_size/sr
    plt.plot(t, signal)
    plt.plot(t_hop_size, rms_values, color='blue')
    print(onsets)
    if len(onsets) > 0:
      plt.vlines(t[onsets], ymin=min(signal), ymax=max(signal), color='red')
    plt.show()
  
  return onsets


class CheckingIfImpulse():
  def __init__(
    self,
    sr = 44100,
    max_energy_threshold = 0.99,
    min_energy_threshold = 0.05,
    min_silence_proportion_long_signal = 0.7,
    min_silence_proportion_short_signal = 0.5,
    ):
    
    """ 
    Class to check if an audio signal contains only impulsive sounds.
    Parameters
    ----------
    sr : int, optional
      Sampling rate of the audio signal. Default is 44100.
    max_energy_threshold : float, optional
      Maximum energy threshold for detecting impulsive sounds. Default is 0.99.
    min_energy_threshold : float, optional
      Minimum energy threshold for detecting impulsive sounds. Default is 0.05.
    min_silence_proportion_long_signal : float, optional
      Minimum proportion of silence required in long signals. Default is 0.7.
    min_silence_proportion_short_signal : float, optional
      Minimum proportion of silence required in short signals. Default is 0.5.
        
    """
    
    self.sr = sr
    self.max_energy_threshold = max_energy_threshold
    self.min_energy_threshold = min_energy_threshold
    self.min_silence_proportion_long_signal = min_silence_proportion_long_signal
    self.min_silence_proportion_short_signal = min_silence_proportion_short_signal
    
  def forward(self, signal, plot_figure=False):
    
    # Only working with mono signals
    if len(signal.shape) == 2 and signal.shape[0] > 1:
      signal = signal.mean(axis=1)

    # signal = signal.reshape(-1, 1)
    signal = signal.flatten()
    
    # We only compute silence proportion for signals longer than 0.5s 
    if len(signal) <= int(0.5*self.sr):  
      return True
  
    silence_proportion = compute_silence_proportion(
        signal=signal,
        sr=self.sr,
        rms=True,
        energy_threshold=self.max_energy_threshold,
        min_threshold=self.min_energy_threshold,
        plot=plot_figure)  
    
    if plot_figure:
      print("Silence proportion: ", silence_proportion)  
  
    if len(signal) > self.sr:
      if silence_proportion < self.min_silence_proportion_long_signal or silence_proportion == 1.:
        # if complete silence, remove the signal
        return False
    else:
      if silence_proportion < self.min_silence_proportion_short_signal or silence_proportion == 1.:
        return False
    
    return True