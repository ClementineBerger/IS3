import numpy as np
import librosa
import math


def linear2dB(x, gain):
  return gain * np.log10(np.abs(x))


def weighting_function(freq):
  """
  A-weighting function used to compute level in dBA.

  Parameters
  ----------
  freq : np.ndarray
      Frequencies
  """
  num = (12194**2) * (freq**4)
  denom = (freq**2) + (20.6**2)
  denom *= (freq**2) + (12194**2)
  denom *= np.sqrt(((freq**2) + (107.7**2)))
  denom *= np.sqrt(((freq**2) + (737.9**2)))
  return num / denom


def a_weightings(freq):
  """
  Normalized weighting function.

  Parameters
  ----------
  freq : np.ndarray
      Frequencies

  Returns
  -------
  float or np.ndarray
      Normalized A-weighting function
  """

  weights = weighting_function(freq) / weighting_function(1000)

  return weights


def compute_rms_from_waveform(x):
  """
  Compute the signal RMS level from the waveform.

  Parameters
  ----------
  x : np.ndarray
    waveform signal

  Returns
  -------
  float
      RMS level
  """
  return np.sqrt(np.mean(x**2))


def compute_rms_from_fft(spectrum):
  """
  Generate the signal RMS level from the fft spectrum.
  Parameters
  ----------
  spectrum : fft spectrum

  Returns
  -------
  float
      RMS level
  """

  spectrum_shapes = spectrum.shape
  if len(spectrum_shapes) == 1:
    spectrum = spectrum.reshape(-1, 1)
  m = spectrum_shapes[0]   # (1+nfft//2)
  
  rms = (1 / (2 * m - 2)) * np.sqrt(
      2 * np.sum(np.abs(spectrum)**2, axis=0) - np.abs(spectrum[0, :])**2
  )
  
  if len(rms) == 1:
    return rms[0]

  return rms


class DBANormalization:
  """
  A class to perform dBA normalization on audio signals.
  audio_duration : float
    The duration of the audio signal in seconds.
    
  Attributes
  ----------
  sr : int
    The sample rate of the audio signal.
  audio_duration : float
    The duration of the audio signal in seconds.
  audio_length : int
    The length of the audio signal in samples.
  frequencies : numpy.ndarray
    The frequency bins for the FFT of the audio signal.
    The A-weighting coefficients for the frequencies.
    
  Methods
  -------
  compute_weights()
    Compute the A-weighting coefficients for the frequencies.
  check_if_mono_and_flatten(audio)
    Check if the audio signal is mono and flatten it if necessary.
  compute_dBA_level(audio)
  scaling_factor(db_init, db_target)
    Compute the scaling factor to achieve a target dBA level.
  compute_dba_normalization_coeff(audio, db_target)
  """
  def __init__(self, audio_duration, sr=44100):
    self.sr = sr
    self.audio_duration = audio_duration
    self.audio_length = int(sr * audio_duration)
    self.frequencies = np.fft.rfftfreq(int(sr * audio_duration), 1 / sr)
    
    self.weights = self.compute_weights()
  
  def compute_weights(self):
    return a_weightings(self.frequencies)
  
  def check_if_mono_and_flatten(self, audio):
    """
    Check if the audio is mono and flatten it if necessary.
    Parameters
    ----------
    audio : numpy.ndarray
      The input audio array. It can be either a 1D array (mono) or a 2D array with shape (1, n) (mono).
      
    Returns
    -------
    numpy.ndarray
      The flattened mono audio array.
      
    Raises
    ------
    ValueError
      If the audio is not mono or if the audio length does not match the expected audio duration.
    """
    
    if len(audio.shape) == 2:
      if audio.shape[0] == 1:
        return audio[0]
      else:
        raise ValueError('Only mono audio is supported')
      
    if len(audio) != self.audio_length:
      raise ValueError('Audio length must be equal to the audio duration')
    return audio
  
  def compute_dBA_level(self, audio):
    """
    Compute the dBA level of an audio signal.
    Parameters
    ----------
    audio : numpy.ndarray
      The audio signal to be analyzed. It should be a 1D or 2D array.
    weights : numpy.ndarray
      The frequency weights to be applied to the audio signal's spectrum.
    Returns
    -------
    dBA_level : float
      The computed dBA level of the audio signal.
    Notes
    -----
    This function applies a Hanning window to the audio signal, computes its
    FFT, applies the given frequency weights, and then calculates the RMS
    value from the weighted spectrum. The dBA level is then computed from
    the RMS value using a linear to dB conversion with a gain of 20.
    """
    
    audio = self.check_if_mono_and_flatten(audio)

    try:
      spectrum = np.fft.rfft(audio * np.hanning(self.audio_length))
    except Exception as e:
      print(self.)
      raise e

    rms = compute_rms_from_fft(spectrum=self.weights * spectrum)

    dBA_level = linear2dB(x=rms, gain=20)

    return dBA_level  

  def scaling_factor(self, db_init, db_target):
    return 10**((db_target - db_init) / 20)
  
  def compute_dba_normalization_coeff(self, audio, db_target):
    """
    Compute the normalization coefficient to achieve a target dBA level for an audio signal.
    
    Parameters
    ----------
    audio : numpy.ndarray
      The input audio signal. Must be a 1D numpy array representing mono audio.
    db_target : float
      The target dBA level to normalize the audio to.
    sr : int, optional
      The sample rate of the audio signal, by default 44100.
    weights : numpy.ndarray, optional
      Precomputed A-weighting coefficients. If None, they will be computed based on the sample rate, by default None.
      
    Returns
    -------
    float
      The normalization coefficient to scale the audio signal to the target dBA level.
    Raises
    ------
    ValueError
      If the input audio is not mono.
    """
    
    init_db_level = self.compute_dBA_level(
        audio=audio)

    norm_coeff = self.scaling_factor(db_init=init_db_level, db_target=db_target)

    return norm_coeff  