""" 
Impulse sound detection algorithme using matching pursuit
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from oct2py import Oct2Py

from scipy.signal import find_peaks

# Load the LTFAT octave package for matching pursuit
oc = Oct2Py()
oc.eval('pkg load ltfat')

class ImpulseDetectionAlgorithm:
  """
  ImpulseDetectionAlgorithm is a class designed for detecting impulse sounds in audio signals. 
  It utilizes multi-resolution Gabor decomposition and matching pursuit algorithms to identify and confirm 
  the presence of impulse sounds.
  """
  def __init__(self,
               sr: int = 16000,
               window_sizes: list = [512, 1024, 2048, 8192],
               hop_sizes: list = [128, 256, 512, 2048],
               onset_hop_size: int = 512,
               onset_threshold: float = 0.2,
               peak_detection_distance_s: float = 0.1,
               peak_prominence=0.3,
               mp_window_size_s: float = 5.,
               number_small_windows: int = 3,
               rms_window_size: int = 1024,
               rms_hop_size: int = 512,
               onset_selection: bool = True,
               onset_selection_window_size_s: float = 1.,
               onset_selection_threshold: float = 0.1,
               ):
    """
    Initialize the impulse sound detection parameters.
    Parameters
    ----------
    sr : int, optional
      Sampling rate, by default 16000.
    window_sizes : list, optional
      List of window sizes for multi-resolution analysis, by default [512, 1024, 2048, 4096, 8192].
    hop_sizes : list, optional
      List of hop sizes corresponding to the window sizes, by default [128, 256, 512, 1024, 2048].
    onset_hop_size : int, optional
      Hop size for onset detection, by default 512.
    onset_threshold : float, optional
      Threshold for onset detection, by default 0.2.
    peak_detection_distance_s : float, optional
      Minimum distance between detected peaks in seconds, by default 0.1.
    peak_prominence : float, optional
      Prominence of peaks for peak detection, by default 0.3.
    mp_window_size_s : float, optional
      Window size for matching pursuit in seconds, by default 5.0.
    number_small_windows : int, optional
      Number of small windows for multi-resolution analysis, by default 3.
    rms_window_size : int, optional
      Window size for RMS envelope calculation in samples, by default 1024.
    rms_hop_size : int, optional
      Hop size for RMS envelope calculation in samples, by default 512.
    onset_selection : bool, optional
      Whether to perform onset selection based on level variation, by default True.
    onset_selection_window_size_s : float, optional
      Window size for onset selection in seconds, by default 1.0.
    onset_selection_threshold : float, optional
      Threshold for onset selection, by default 0.1.
    **kwargs : dict
      Additional keyword arguments.
    """
    self.sr = sr
    
    # Multi-resolution Gabor parameters
    self.window_sizes = window_sizes
    self.hop_sizes = hop_sizes
    
    # Onset detection parameters
    self.onset_hop_size = onset_hop_size
    self.onset_threshold = onset_threshold
    

    self.mp_window_size_s = mp_window_size_s
    
    # Selection of onsets : removing onsets that are from a volume augmentation
    # rather than a real onset
    self.onset_selection = onset_selection
    self.onset_selection_window_size_s = onset_selection_window_size_s
    self.onset_selection_threshold = onset_selection_threshold  
    
    # RMS envelope parameters
    self.number_small_windows = number_small_windows
    self.rms_window_size = rms_window_size
    self.rms_hop_size = rms_hop_size
  
    
    # Peak detection parameters (on the gabor coefficients)
    self.peak_detection_distance_s = peak_detection_distance_s
    self.peak_prominence = peak_prominence
    
    multi_gabor = []
    for i in range(len(self.hop_sizes)):
      multi_gabor.append('blackman')
      multi_gabor.append(self.hop_sizes[i])
      multi_gabor.append(self.window_sizes[i])
      
    self.multi_gabor = tuple(multi_gabor)
    
  def rms_envelope(self, signal):
    """
    Compute the root mean square (RMS) envelope of an audio signal.
    Parameters
    ----------
    signal : np.ndarray
      The input audio signal.
    Returns
    -------
    np.ndarray
      The RMS envelope of the input signal.
    """
    
    rms_env = librosa.feature.rms(y=signal, 
                                  frame_length=self.rms_window_size, 
                                  hop_length=self.rms_hop_size, 
                                  center=True)
    return rms_env
    
  def matching_pursuit(self, signal):
    """
    Perform matching pursuit on the given signal using multi Gabor atoms.
    Parameters
    ----------
    signal : array_like
      The input signal to be analyzed.
    Returns
    -------
    coeffs : list
      A list of coefficients obtained from the matching pursuit algorithm.
    """
    
    res = oc.feval(
        'multidgtrealmp', signal,
        self.multi_gabor,
        nout=1, verbose=False
    )

    coeffs = [res[i, 0] for i in range(res.shape[0])]    
    
    return coeffs  # [freq, time]
  
  def onset_detection(self, signal):
    """
    Detects the onset times in an audio signal.
    Parameters
    ----------
    signal : np.ndarray
      The audio signal from which to detect onsets.
    Returns
    -------
    onsets : np.ndarray
      Array of sample indices where onsets are detected.
    """
    
    onset_envelope = librosa.onset.onset_strength(y=signal, sr=self.sr, hop_length=self.onset_hop_size)
    onsets = librosa.onset.onset_detect(
        y=signal,
        sr=self.sr,
        hop_length=self.onset_hop_size,
        units='samples',
        normalize=True,
        onset_envelope=onset_envelope,
        delta=self.onset_threshold)
    
    return onsets
  
  def check_level_variation(self, rms_env, onsets, signal):
    """
    Check the level variation of the RMS envelope and remove onsets that do not show significant variation.
    Parameters
    ----------
    rms_env : ndarray
      The RMS envelope of the signal.
    onsets : ndarray
      Array of detected onsets.
    signal : ndarray
      The original audio signal.
    Returns
    -------
    ndarray
      Array of onsets after removing those with insignificant level variation.
    """
    
    window_size_s = self.onset_selection_window_size_s #s
    window_size = int(window_size_s * self.sr/self.rms_hop_size)
    
    # quantile 0.75 of the rms envelope
    eps = np.quantile(abs(signal),q=0.85) * self.onset_selection_threshold
    
    # si pendant une seconde, le niveau rms reste globalement constant, ou augmente légèrement, l'onset est retiré
    onsets_to_remove = []
    for onset in onsets:
      rms_window = rms_env[0][int(onset/self.rms_hop_size) : int(onset/self.rms_hop_size) + window_size]
      max_begin_window = np.max(rms_window[0:int(0.1 * self.sr/self.rms_hop_size)])
      if abs(max_begin_window - np.median(rms_window)) < eps:
        onsets_to_remove.append(onset) 
      
    onsets = np.setdiff1d(onsets, onsets_to_remove)
    return onsets
      

  
  def peak_detection(self, gabor_coeffs, num_win=3):
    """
    Detects peaks in the Gabor coefficients.
    Parameters
    ----------
    gabor_coeffs : list of np.ndarray
      List of Gabor coefficients for different window sizes.
    num_win : int, optional
      Number of windows to consider for peak detection, by default 3.
    Returns
    -------
    peaks : np.ndarray
      Indices of the detected peaks.
    sum_small_win : np.ndarray
      Sum of the absolute values of the upsampled coefficients for the first `num_win` windows along the time axis.
    Notes
    -----
    The function performs the following steps:
    1. Upsamples the Gabor coefficients for the first `num_win` windows.
    2. Computes the sum of the absolute values of the upsampled coefficients along the time axis.
    3. Detects peaks in the resulting sum using the specified distance and prominence criteria.
    """    
    
    # upsampling des coeffs
    coeffs_upsampled = []
    for j in range(num_win):
      upsampled_coeff = np.repeat(
          gabor_coeffs[j],
          self.window_sizes[j] //
          self.window_sizes[0],
          axis=1)
      coeffs_upsampled.append(upsampled_coeff)

    # sum of the coeffs for the first num_win windows along the time axis
    # sum_small_win = 0.
    # for coeff in coeffs_upsampled[0:num_win]:
    #   sum_small_win += np.sum(np.abs(coeff), axis=0)

    # sum_small_win = sum_small_win / num_win
    
    # for all time indexes, the max of the three windows coefficients
    sum_coeffs_upsampled = [np.sum(np.abs(coeff), axis=0) for coeff in coeffs_upsampled[0:num_win]]
    sum_small_win = np.max(np.stack(sum_coeffs_upsampled), axis=0)
    
    # Peak detection on the obtained sum
    distance = int(self.peak_detection_distance_s * self.sr / self.hop_sizes[0])
    height = np.max(gabor_coeffs[-1])
    
    prominence = np.max(sum_small_win) * self.peak_prominence
    
    peaks, _ = find_peaks(
      sum_small_win,
      height=height,
      distance=distance,
      prominence=prominence)
    
    return peaks, sum_small_win
  
  def impulse_detection(self, signal, plot_figure=False):
    
    # Onset detection
    onsets = self.onset_detection(signal)
    rms_env = self.rms_envelope(signal)
    if self.onset_selection :
      onsets = self.check_level_variation(rms_env, onsets, signal)
    onset_times = onsets / self.sr
    
    if plot_figure:
      plt.figure(0, figsize=(6, 3))
      plt.plot(np.arange(len(signal))/self.sr, signal)
      plt.plot(np.arange(len(rms_env[0]))*self.rms_hop_size/self.sr, rms_env[0], color='b', label='RMS envelope')
      plt.vlines(
        onset_times,
        ymin=np.min(signal),
        ymax=np.max(signal),
        color='r',
        linestyle='--',
        label='Onsets')
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')
      plt.title('Signal with detected onsets')
      
    min_distance_between_onsets = self.mp_window_size_s / 3
    
    impulse_idx = []
    is_impulse = False
    
    for i in range(0, len(onset_times)):
      if i > 0 and onset_times[i] - onset_times[i - 1] < min_distance_between_onsets:
        continue

      if (onsets[i] - int(self.mp_window_size_s*self.sr/2)) < 0:
        start_ind = 0.
        analysis_signal = signal[0:onsets[i] + int((self.mp_window_size_s) * self.sr)]
      elif (onsets[i] + int(self.mp_window_size_s*self.sr/2)) > len(signal):
        start_ind = len(signal) - int(self.mp_window_size_s * self.sr)
        analysis_signal = signal[len(signal) - int(self.mp_window_size_s * self.sr):]
      else:
        start_ind = onsets[i] - int(self.mp_window_size_s*self.sr/2)
        analysis_signal = signal[onsets[i] - int(self.mp_window_size_s*self.sr/2):onsets[i] + int(self.mp_window_size_s*self.sr/2)]
      
      # finding the indexes of the onsets in the window 
      onsets_in_window = (onset_times >= start_ind / self.sr) & (onset_times <= (start_ind/self.sr + self.mp_window_size_s))
        
      # Matching pursuit
      gabor_coeffs = self.matching_pursuit(analysis_signal)
      peaks, sum_small_win = self.peak_detection(gabor_coeffs, num_win=self.number_small_windows)
      
      if len(peaks) != 0:
        # as the window is 5s long, several impulses can be detected, or an impulse can be 
        # detected but it is not the one corresponding to the onset.
        # We only store an impulse sufficiently close to the onset
        
        # find the closest peak to the onset:
        # t = np.arange(0, len(analysis_signal), self.hop_sizes[0]) / self.sr + start_ind / self.sr
        t = np.arange(0, gabor_coeffs[0].shape[1]) * self.hop_sizes[0] / self.sr + start_ind / self.sr
        difference_matrix = np.abs(t[peaks][:, np.newaxis] - onset_times[onsets_in_window])
        closest_peak_idx = np.argmin(difference_matrix, axis=0)

        # check if the closest peak is close enough to the onset (less than 0.2s)
        for j in range(np.sum(onsets_in_window)):
          if np.abs(t[peaks[closest_peak_idx[j]]] - onset_times[onsets_in_window][j]) < 0.2:
            impulse_idx.append(onsets[onsets_in_window][j]) 
            is_impulse = True

      
      if plot_figure:
        fig, axs = plt.subplots(1, 2, figsize=(12, 3))
        
        # Left subplot: current figure
        t = np.arange(0, gabor_coeffs[0].shape[1]) * self.hop_sizes[0] / self.sr + start_ind / self.sr
        axs[0].plot(t, sum_small_win[:len(t)])
        axs[0].vlines(onset_times[onsets_in_window], 
                ymin=0.,
                ymax=np.max(sum_small_win),
                color='r', linestyle='--')
        axs[0].plot(t[peaks], sum_small_win[:len(t)][peaks], 'kx')
        if len(peaks) > 0:
          axs[0].plot(t[peaks[closest_peak_idx]], sum_small_win[:len(t)][peaks[closest_peak_idx]], 'rx')
        axs[0].set_ylabel('Mean of the sum of the coefficients')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_title('Analysis window around onset')
        
        # Right subplot: empty figure
        for j in range(len(gabor_coeffs)):
          t_win = np.arange(0, len(analysis_signal), self.hop_sizes[j]) / self.sr + start_ind / self.sr
          axs[1].plot(t_win, np.sum(np.abs(gabor_coeffs[j]), axis=0)[:len(t_win)], label="M = " + str(self.window_sizes[j]))
        axs[1].plot(t, sum_small_win[:len(t)], label="Sum of the coefficients", color='k')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].set_ylabel('Sum of the coefficients')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_title("Max coeff large window = " + str(np.max(np.abs(gabor_coeffs[-1]))))
        
        plt.tight_layout()
        
    # remove duplicates in impulse_idx
    impulse_idx = list(set(impulse_idx))
    # sort from the earliest to the latest onset
    impulse_idx.sort()
    
    window_around_impulse = int(1. * self.sr)
    half_window = window_around_impulse // 2
    impulse_windows = []
    for idx in impulse_idx:
      start = max(0, idx - half_window)  # Prevent negative start index
      end = min(len(signal), idx + half_window)  # Prevent end index beyond signal length
      impulse_windows.append([start, end])      
    
    if len(impulse_windows) > 0:
      merge_distance = int(5 * self.sr)  # 5 seconds
      merged_impulse_windows = []
      current_start, current_end = impulse_windows[0]  # Initialize with the first window

      for start, end in impulse_windows[1:]:
        # Check for overlap or proximity between the current window and the next one
        if start <= current_end + merge_distance:
          # Extend the current window to include the next one
          current_end = max(current_end, end)
        else:
          # Add the current window to the results and move to the next one
          merged_impulse_windows.append([current_start/self.sr, current_end/self.sr]) # convert to seconds to use with different sampling rates
          current_start, current_end = start, end

      # Add the last window
      merged_impulse_windows.append([current_start/self.sr, current_end/self.sr])
    
    else: 
      merged_impulse_windows = []
        
    if plot_figure:
      plt.figure(len(onset_times)+2, figsize=(6, 3))
      plt.plot(np.arange(len(signal))/self.sr, signal)
      plt.vlines(
        np.array(impulse_idx)/self.sr,
        ymin=np.min(signal),
        ymax=np.max(signal),
        color='lightgreen',
        linestyle='--',
        label='Onsets')
      for start_time, end_time in merged_impulse_windows:
        plt.axvspan(start_time, end_time, color='g', alpha=0.5)
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')
      plt.title('Confirmed detected onsets')   
    
    return is_impulse, np.array(impulse_idx)/self.sr, merged_impulse_windows
