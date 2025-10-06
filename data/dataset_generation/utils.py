from datetime import datetime
import os
import random
import numpy as np
import audioread
import soundfile
import librosa
import config
import snr

from scipy.stats import truncnorm

from py_salt.event_mapping import EventExplorer
from scene_mapping import SceneExplorer

def import_audio_file(file_path,
  sr=None,
  res_type='kaiser_best',
  mono=False,
  offset=0,
  duration=None,
  print_warning=True,
  return_offset=False):


  # --- Handle portion loading
  with audioread.audio_open(file_path) as f:
    file_duration = f.duration
    native_sr = f.samplerate  # get native sampling rate if sr is None

  if sr is None:
    sr = native_sr

  # Convert duration to samples if specified
  duration_samples = int(duration * sr) if duration is not None else None

  if duration is not None and duration > file_duration:
    if print_warning:
      print('Selected duration is greater than the file total duration,'
                    ' loading the entire file...')
    offset_samples = 0
    duration_samples = int(file_duration * sr)
  else:
    max_offset_samples = max(0, int(file_duration * sr) -
                                   (duration_samples or 0))
    if offset >= 0:
      offset_samples = offset
      if offset_samples + (duration_samples or 0) > int(file_duration * sr):
        if print_warning:
          print('Audio selection goes beyond the file limits, '
                'resetting offset to 0.')
        offset_samples = 0
    else:
      offset_samples = np.random.randint(0, max_offset_samples + 1)

  # --- Select "optimal" lib to import the audio file
  use_librosa = True
  try:
    audio_format = soundfile.info(file_path).format
    if audio_format == 'FLAC':
      use_librosa = False
  except RuntimeError:
    pass

  if use_librosa:
    sig, sr = librosa.load(
      file_path, sr=sr, res_type=res_type, mono=mono,
      offset=offset_samples / sr,
      duration=duration_samples / sr if duration_samples else None)

    if sig.ndim == 1:
      sig = np.expand_dims(sig, 0)
  else:
    start = offset_samples
    stop = start + duration_samples if duration_samples else None
    sig, _ = soundfile.read(file_path,
                            start=start,
                            stop=stop,
                            dtype='float32',
                            always_2d=True)
    sig = sig.T
    if mono:
      sig = np.expand_dims(np.mean(sig, axis=0), 0)

  # --- Check for empty signal before resampling
  if sig.size == 0:
    raise ValueError(f'Loaded audio is empty for file {file_path}. '
                     'Cannot proceed with processing.')

  # --- Resample if necessary
  if sr != native_sr and use_librosa is False:
    sig = librosa.resample(
      y=sig, orig_sr=native_sr, target_sr=sr, res_type=res_type)

  if return_offset:
    return sig, sr, offset_samples
  else:
    return sig, sr


def normalize_signal(sig):
  """Normalize a signal to the range [-1, 1].

  Parameters
  ----------
  sig : np.array
      Input signal of shape (n_channels, n_samples)

  Returns
  -------
  np.arrays
      _description_
  """
  global_max = np.max(np.abs(sig))
  gain = 1 / global_max if global_max != 0 else 1
  normalized_signal = sig * gain
  return normalized_signal, gain

def choose_value_uniform():
  random.seed(datetime.now().timestamp())
  return random.uniform(0,1)

def initialize_truncated_gaussian(mean, std, min_val, max_val):
  a = (min_val - mean) / std
  b = (max_val - mean) / std

  distribution = truncnorm(
      a, b, loc=mean, scale=std)

  return distribution  


def sample_from_trunc(mean, std):
  while True:
    random.seed(datetime.now().timestamp())
    value = random.gauss(mean, std)
    if mean - 2*std <= value <= mean + 2*std:
      return round(value, 2)

def remove_signal_edge_silence(sig, threshold=1e-7):
  """
  Remove silence at left and right edges of a signal

  Parameters
  ----------
  sig: numpy array with shape (n_channels, n_samples_ini)
     waveform of the input signal

  threshold: float (default 1e-7)
    threshold of the energy normalized cumulative sum used to detect
    silence durations

  Returns
  -------
  numpy array with shape (n_channels, n_samples_out)
     waveform of the input signal with start and end silences removed
  """

  energy = np.mean(np.square(sig), axis=0)

  left_cumsum = np.cumsum(energy)
  left_cumsum /= np.max(left_cumsum)
  left_cut_idx = np.sum(left_cumsum < threshold)

  right_cumsum = np.cumsum(energy[::-1])
  right_cumsum /= np.max(right_cumsum)
  right_cut_idx = len(energy) - np.sum(right_cumsum < threshold)

  #   plt.figure()
  #   plt.plot(energy/np.max(energy))
  #   plt.plot(left_cumsum, 'r')
  #   plt.plot(right_cumsum[::-1], 'k')
  #   plt.stem([left_cut_idx], [1.], 'r')
  #   plt.stem([right_cut_idx], [1.], 'k')

  return sig[:, left_cut_idx:right_cut_idx]


def apply_fade(audio_waveform, sample_rate, fade_in_time, fade_out_time):
    """
    Applique un fade-in et un fade-out à une waveform mono.
    
    :param audio_waveform: np.ndarray, waveform audio mono
    :param sample_rate: int, fréquence d'échantillonnage (en Hz)
    :param fade_in_time: float, durée du fade-in (en millisecondes)
    :param fade_out_time: float, durée du fade-out (en millisecondes)
    :return: np.ndarray, waveform avec fade appliqué
    """
    
    # Conversion des durées en nombre d'échantillons
    fade_in_samples = int(fade_in_time  * sample_rate)
    fade_out_samples = int(fade_out_time * sample_rate)
    
    # Création de courbes de fade
    fade_in_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in_samples)))
    fade_out_curve = 0.5 * (1 - np.cos(np.linspace(np.pi, 2 * np.pi, fade_out_samples)))
    
    # Application du fade-in
    audio_waveform[...,:fade_in_samples] *= fade_in_curve
    
    # Application du fade-out
    audio_waveform[...,-fade_out_samples:] *= fade_out_curve
    
    return audio_waveform


class SaltWrapper():
  """Wrapper class of py-salt
  """
  _instance = None # Class atrribute to hold the single instance

  @property
  def event_mapper(self):
    return self._event_mapper

  @property
  def scene_mapper(self):
    return self._scene_mapper

  def __new__(cls, *args, **kwargs):
    # Check if an instance already exists
    if cls._instance is None:
      # Create a nnew instance if none exists
      cls._instance = super(SaltWrapper, cls).__new__(cls)
    return cls._instance

  def __init__(self):
    if not hasattr(self, '_initialized'):
      assert os.path.isfile(config.IMPULSES_EVENT_MAP_FP), \
        f'{config.IMPULSES_EVENT_MAP_FP} does not exist'

      self._event_map_file_path = config.IMPULSES_EVENT_MAP_FP
      self._event_roots_map_file_path = config.IMPULSES_ROOT_FP

      self._scene_map_file_path = config.SCENES_MAP_FP
      self._scene_roots_map_file_path = config.SCENES_ROOT_FP

      self.impulse_labels = None

      self._event_mapper = EventExplorer(
        map_md_file_path=self._event_map_file_path,
        roots_md_file_path=self._event_roots_map_file_path
      )

      self._scene_mapper = SceneExplorer(
        map_md_file_path=self._scene_map_file_path,
        roots_md_file_path=self._scene_roots_map_file_path
      )

      self._initialized = True

      self.init_impulse_labels()

  def init_impulse_labels(self):
    self.impulse_labels = self._event_mapper.map_df[
      'standard_event'].unique().tolist()



def compute_impulse_gain_for_unbalanced_mix(impulse, background, sr, target_snr):
  # IMPORTANT NOTE: compute_background_gain_for_unbalanced_mix computes
  # a gain for the background signal based on the reference signal so that
  # dB_ref = dBA_bkg + target_snr. If we want to change the amplitude of
  # the impulse, we need to pass it as the "background_signal" and reverse
  # the SNR.
  return snr.compute_background_gain_for_unbalanced_mix(
    reference_signal=background,
    background_signal=impulse,
    only_mask_with_background=True,
    sampling_rate=sr,
    target_snr=-target_snr
  )
