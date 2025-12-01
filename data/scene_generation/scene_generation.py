"""Scene generation classes
"""
from matplotlib import pyplot as plt
from audiomentations import Compose, ApplyImpulseResponse
from dba_normalization import DBANormalization

import os
import json
import random
import secrets
import numpy as np
import pandas as pd
import soundfile as sf

import scipy.signal as sig

import utils
import config
import data_loader

from tqdm import tqdm

# Initialize distributions

snr_distrib = {}
for scene, snr_params in config.SNR_RANGE.items():
  snr_distrib[scene] = utils.initialize_truncated_gaussian(**snr_params)

scene_dba_distrib = {}
for scene, dba_params in config.SCENE_DBA.items():
  scene_dba_distrib[scene] = utils.initialize_truncated_gaussian(**dba_params)


class Generator:
  """Class to generate multiple scenes
  """
  def __init__(self,
               root_dir: str,
               sr: int = 44100,
               duration: float = 5,
               irs_file_path_list: list = None,
               impulse_max_duration: float = 2.0,
               n_backgrounds: int = 1,
               max_n_impulses: int = 1,
               background_scene=None,
               impulse_type=None, 
               subset="train", # train, val, test
               ):
    """Generator class to generate multiple scenes of mixtures with
    backgrounds and impulses.

    Parameters
    ----------
    root_dir : str
        Directory to store the generated scenes
    sr : int, optional
        sample rate, by default 44100
    duration : float, optional
        duration of each scene, by default 10
    impulse_max_duration : float, optional
        max impulse duration, by default 2.0
    n_backgrounds : int, optional
        num of backgrounds, by default 1
    n_impulses : int, optional
        num of impulses, by default 1
    background_scene : _type_, optional
        background scene label, by default None
    impulse_type : str, optional
        impulse type. Can be 'natural' or 'synthetic'. Can be also None if the
        scene has no impulses, by default 'natural'
    """

    self.root_dir = root_dir
    self.subset = subset
    
    self.foreground_loader = data_loader.ForegroundLoader()
    self.background_loader = data_loader.BackgroundLoader()
    self.ir_loader = data_loader.IrLoader()

    self.sr = sr
    self.duration = duration
    self.impulse_max_duration = impulse_max_duration
    self.n_backgrounds = n_backgrounds
    self.max_n_impulses = max_n_impulses
    self.impulse_type = impulse_type
    self.background_scene = background_scene
    self.irs_file_path_list = irs_file_path_list

    self.log_file = os.path.join(self.root_dir, 'log.txt')
    self.log = []
    
    assert self.max_n_impulses == config.N_IMPULSES_DISTRIB['max_val'], 'max_n_impulses should be equal to config.N_IMPULSES_DISTRIB[\'max_val\']'
    
    self.n_impulses_distrib = utils.initialize_truncated_gaussian(
      **config.N_IMPULSES_DISTRIB)

  def generate(self, n_scenes, start_file_ind=None, raise_exec=False):
    # start_file_ind : for multi CPU generation, to avoid overwriting files
    
    for i in tqdm(range(n_scenes)):
      # Random number of impulsion
      n_impulses = int(self.n_impulses_distrib.rvs())
      
      scene_index = start_file_ind + i if start_file_ind is not None else i
      
      if scene_index is not None:
        num_folder = f"{scene_index//1000:03d}"
      
        audio_output_dir = os.path.join(
          self.root_dir,
          "audio",
          self.subset,
          num_folder,
          )
        
        json_output_dir = os.path.join(
          self.root_dir,
          "audio_info",
          self.subset,
          num_folder,
          )
      
      else: 
        audio_output_dir = os.path.join(
          self.root_dir,
          "audio",
          self.subset,
          )
        
        json_output_dir = os.path.join(
          self.root_dir,
          "audio_info",
          self.subset,
          )        
      
      try:
        scene = Scene(
          audio_output_dir=audio_output_dir,
          json_output_dir=json_output_dir,
          backgrounds_pdf = self.background_loader.pdf_metadata,
          impulses_pdf = self.foreground_loader.pdf_metadata,
          irs_file_path_list = self.irs_file_path_list,
          sr = self.sr,
          duration = self.duration,
          impulse_max_duration = self.impulse_max_duration,
          n_backgrounds = self.n_backgrounds,
          n_impulses = n_impulses,
          background_scene = self.background_scene,
          impulse_type=self.impulse_type,
          scene_index=scene_index
        )
        scene.generate_scene()
      except Exception as e:
        error_message = f'Error in scene {i+1}: {str(e)}'
        print(error_message)
        self.log.append(error_message)
        with open(self.log_file, mode='a', encoding='utf-8') as log_f:
          log_f.write(error_message + '\n')
        if raise_exec:
          raise e


class Scene():
  """Class to generate a scene (background(s) + impulse(s))
  """
  @property
  def metadata(self):
    metadata = {}
    for i, background in enumerate(self.backgrounds):
      metadata[f'background_{i+1}'] = background.metadata
    for i, impulse in enumerate(self.impulses):
      metadata[f'impulse_{i+1}'] = impulse.metadata

    # Update with scene-specific metadata
    metadata.update({
      'audio_root_dir': self.audio_output_dir,
      'json_root_dir': self.json_output_dir,
      'background_scene': self.background_scene,
      'has_impulses': self.n_impulses > 0,
      'n_impulses': self.n_impulses,
      'impulse_max_duration': self.impulse_max_duration,
      'impulse_type': self.impulse_type,
      'normalization_gain': self.normalization_gain,
      'dba_target': self.dba_target,
      'dba_normalization_coeff': self.dba_normalization_coeff,
      'ir_file_path': self.ir_file_path,
    })
    
    # Add extra metadata
    metadata.update(self._extra_metadata)    

    return metadata

  def __init__(self,
               audio_output_dir: str,
               json_output_dir: str,
               backgrounds_pdf: pd.DataFrame,
               impulses_pdf: pd.DataFrame,
               irs_file_path_list: list = None,
               sr: int = 44100,
               duration: float = 10,
               impulse_max_duration: float = 2.0,
               n_backgrounds: int = 1,
               n_impulses: int = 1,
               background_scene=None,
               impulse_type=None,
               scene_index=None,
               ):
    # if os.path.isdir(output_dir):
    #   raise IsADirectoryError(f'Directory {output_dir} already exists!')

    assert sr in [16000, 32000, 44100, 48000], 'Not acceptable sample rate'

    assert impulse_type in ['synthetic', 'natural', None]

    self.audio_output_dir = audio_output_dir
    self.json_output_dir = json_output_dir
    self.scene_index = scene_index 

    self.backgrounds_pdf = backgrounds_pdf
    self.impulses_pdf = impulses_pdf

    if irs_file_path_list:
      self.irs_pdf = pd.DataFrame({'file_path': irs_file_path_list})
    else:
      self.irs_pdf = None

    if self.irs_pdf is not None:
      self.ir_file_paths = self.get_ir_file_paths()
    else: 
      self.ir_file_paths = None

    self.sr = sr
    self.duration = duration
    self.impulse_max_duration = impulse_max_duration

    self.background_scene = background_scene
    self.n_backgrounds = n_backgrounds
    self.background_mixture = None

    self.n_impulses = n_impulses
    self.impulse_type = impulse_type
    self.impulse_mixture = None

    self.backgrounds = []
    self.impulses = []

    self.dba_target = None
    self.normalization_gain = None
    self.dba_normalization_coeff = None
    
    self.ir_file_path = None
    self.rt_60 = None
    self.ir_type = None
    
    self.dba_normalization = DBANormalization(self.duration, self.sr)

    self._init_scene_directory()
    
    self._extra_metadata = {}

  def update_metadata(self, new_metadata):
      """Met à jour les métadonnées de la scène."""
      self._extra_metadata.update(new_metadata)    


  def _init_scene_directory(self):
    # if os.path.isdir(self.audio_output_dir):
    #   raise IsADirectoryError(f'"{self.audio_output_dir} is already a directory"')
    os.makedirs(self.audio_output_dir, exist_ok=True)
    
    # if os.path.isdir(self.json_output_dir):
    #   raise IsADirectoryError(f'"{self.json_output_dir} is already a directory"')
    os.makedirs(self.json_output_dir, exist_ok=True)
    
    # as we generate data with multi_cpu, we will use several times the same directory

  def _load_backgrounds(self):
    """Initialize backgrounds. Randomly selects an acoustic scene label with
    sufficient number of data from SALT, then initiates Background class
    instances and adds them in self.backgrounds list.
    """
    df = self.backgrounds_pdf.copy()

    # Step 1: Filter out unwanted scenes
    if self.background_scene is None:
      df = df.loc[df['std_label'].isin(config.BACKGROUND_SCENES)]
    else:
      if self.background_scene not in df['std_label'].unique():
        raise ValueError(f'"{self.background_scene}" not found in the '
                         'background scenes. Available scenes: '
                         f'{df["std_label"].unique().tolist()}')
      else:
        df = df.loc[df['std_label'] == self.background_scene]

    # Step 2: Filter out files with less thatn self.duration
    df = df.loc[df['duration'] >= self.duration]

    # Step 3: Get value counts of std_label (n files per scene)
    value_counts = df['std_label'].value_counts()
    median_count = int(value_counts.median())
    
    # Step 3: for each std label, sample np.minimum(median_count, value_counts) elements
    df = df.groupby('std_label').apply(lambda x: x.sample(np.minimum(median_count, len(x))))

    # Step 4: Filter values where the count is greater than N backgrounds
    new_value_counts = df['std_label'].value_counts()
    valid_scenes = new_value_counts[new_value_counts >= self.n_backgrounds].index

    # Step 5: Randomly choose an acoustic scene
    self.background_scene = secrets.choice(valid_scenes)

    # Get n_backgrounds files for this scene
    df = df.loc[df['std_label'] == self.background_scene].sample(n=self.n_backgrounds)

    for _, row in df.iterrows():
      background = Background(std_label=row['std_label'],
                           file_path=row['file_path'],
                           sr=self.sr,
                           duration=self.duration,
                           mono=True,
                           augment=True if self.irs_pdf else False,
                           ir_file_paths=self.ir_file_paths)

      self.backgrounds.append(background)

    background_mixture = np.mean([t.data for t in self.backgrounds], axis=0)

    # Normalize in dBA
    self.dba_target = scene_dba_distrib[self.background_scene].rvs()

    # Calculate and store dBA normalization coefficient based on the selected 
    # dBA target level
    dba_normalization_coeff = self.dba_normalization.compute_dba_normalization_coeff(
      audio=background_mixture, db_target=self.dba_target)
    

    self.dba_normalization_coeff = dba_normalization_coeff
    self.background_mixture = background_mixture


  def _load_impulses(self):
    """Initialize impulses. Randomly selects an acoustic scene label with
    sufficient number of data from SALT, then initiates Background class
    instances and adds them in self.backgrounds list.
    """
    # --- Get n_impulses files

    # Rule out big files and unwanted scenes
    
    if self.impulse_type is None:
      df = self.impulses_pdf.loc[
      (self.impulses_pdf['duration'] <= self.duration) 
      ].reset_index(drop=True)
      
    else:
      df = self.impulses_pdf.loc[
        (self.impulses_pdf['duration'] <= self.duration) &
        # (self.impulses_pdf['std_label'].isin(config.IMPULSE_EVENTS)) &   #already filtered in mapping
        (self.impulses_pdf['impulse_type'] == self.impulse_type)
      ].reset_index(drop=True)

    # Step 1: Calculate value counts of std_label
    value_counts = df['std_label'].value_counts()

    # Step 2: Find the minimum count M
    # min_count = value_counts.min()
    median_count = int(value_counts.median())

    # Step 3: for each std label, sample np.minimum(median_count, value_counts) elements
    df = df.groupby('std_label').apply(lambda x: x.sample(np.minimum(median_count, len(x))))

    # Step 4: finally, sample n_impulses
    df = df.sample(self.n_impulses)

    df = df.reset_index(drop=True)

    total_duration = 0
    for i, row in df.iterrows():

      # Initiate impulse object
      impulse = Impulse(
        std_label=row['std_label'],
        file_path=row['file_path'],
        sr=self.sr,
        duration=None,
        mono=True,
        impulse_snr=snr_distrib[self.background_scene].rvs(),
        augment=config.IMPULSE_AUGMENT,
        ir_file_paths=self.ir_file_paths,
      )

      # remove edge_silence -> already done in dataset only_impulses
      impulse.data = utils.remove_signal_edge_silence(impulse.data, threshold=np.max(np.abs(impulse.data))*1e-5)

      # Impulse duration check
      if impulse.duration > self.impulse_max_duration:
        # Choose a random segment of the impulse of length self.impulse_max_duration
        # start_idx = random.randint(0, impulse.data.shape[1] - int(self.impulse_max_duration * self.sr))
        start_idx = 0
        impulse.data = impulse.data[:, start_idx:start_idx + int(self.impulse_max_duration * self.sr)]
        # add slight fade in and fade out (10ms)
        impulse.data = utils.apply_fade(impulse.data, self.sr, fade_in_time=0.0, fade_out_time=0.01)
        impulse.duration = self.impulse_max_duration

      # Update total duration
      total_duration += impulse.duration

      if total_duration > self.duration:
        print(f'WARNING: Total duration of impulses ({total_duration}) '
              f'excceds scene\'s duration ({self.duration}). Stopping at '
              f'{i} impulses.')

        self.n_impulses = i
        return

      # Append impulse to self.impulses list
      self.impulses.append(impulse)


  def _get_impulses_idx_on_mixture(self):
    """Create a mixture of audio files by placing them in a
    predefined duration without overlapping. First, impulses are placed
    one after the other in the mixture. Then, the silent part at the end is
    divided in N (N = num of impulses) parts. The silent parts are
    randomly placed between the impulses. The smallest silent part is
    always placed at the beggining of the mixture to avoid overlapping. 

    Returns
    -------
    np.array (n_channels, n_samples)
      Mixture audio of shape (channels, duration_in_samples).
    """
    audio_arrays = [impulse.data for impulse in self.impulses]

    # Determine the output shape
    mixture_samples = int(self.duration * self.sr)

    # Calculate the total duration of all impulses
    total_impulse_samples = sum(audio.shape[-1] for audio in audio_arrays)

    # Ensure the total duration of impulses does not exceed the mixture duration
    if total_impulse_samples > mixture_samples:
      raise ValueError(f'Total samples of impulses ({total_impulse_samples}) '
                       f'exceeds the mixture samples ({mixture_samples}).')

    # Calculate the remaining silent duration
    remaining_silent_duration = mixture_samples - total_impulse_samples

    # Separate the remaining silent part into N randomly sized parts
    n_silent_parts = len(audio_arrays) + 3 # so that if only 1 impulse, it won't be placed at the end
    silent_parts = np.random.dirichlet(
      np.ones(n_silent_parts)) * remaining_silent_duration

    silent_parts = [int(part) for part in silent_parts]
    
    impulse_segments = [['impulse', audio.shape[-1]] for audio in audio_arrays]
    silent_segments = [['silent', silent_part] for silent_part in silent_parts]
    
    # Interleave silent and impulse segments
    all_segments = [segment for pair in zip(silent_segments, impulse_segments) for segment in pair]
    #add one silence at the end
    all_segments.append(['silent', silent_parts[len(impulse_segments)]])
      
    remaining_silent = silent_segments[len(impulse_segments)+1:]
    # Place the remaining segment at random position in all_segments
    for silent in remaining_silent:
      idx = random.randint(0, len(all_segments))
      all_segments.insert(idx, silent)
    
    # Update impulse index
    current_pos = 0
    id_impulse = 0
    for i in range(len(all_segments)):
      if all_segments[i][0] == 'impulse':
        self.impulses[id_impulse].idx = current_pos
        id_impulse += 1
        current_pos += all_segments[i][1]
      else:
        current_pos += all_segments[i][1]


  def _adjust_impulses_amplitude_for_snr(self):
    """Adjust each impule's amplitude to match the target snr between
    each impulse and the background. The impulse is adjusted after its
    position in the mixture is defined. Therefore, the
    _get_impulses_idx_on_mixture function must be called before the
    gains are calculated.

    Raises
    ------
    ValueError
        If the impulses' idxs are None
    """
    for impulse in self.impulses:
      # Add all backgrounds as reference signal (if many)
      if impulse.idx is None:
        raise ValueError('Impulse index in the mixture is None.')

      # Keep only the part of the background that aligns with the impulse
      background_at_impulse = self.background_mixture[
        :, impulse.idx : impulse.idx + impulse.data.shape[1]
      ]

      # Compute gain
      gain = utils.compute_impulse_gain_for_unbalanced_mix(
        impulse=impulse.data,
        background=background_at_impulse,
        sr=self.sr,
        target_snr=impulse.snr
      )

      impulse.data *= gain


  def _create_impulses_mixture(self):
    """Create impulses mixture. After the calculation of each impulse's
    index on the mixture and the adjustment of its amplitude to match
    the target snr in the given index, this function creates a list of
    arrays, each one containing 1 impulse and having the length of the
    mixture.

    Raises
    ------
    ValueError
        If the number of channels between impulses differ
    """
    audio_arrays = [impulse.data for impulse in self.impulses]

    # Determine the output shape
    mixture_samples = int(self.duration * self.sr)
    num_channels = audio_arrays[0].shape[0]

    # Initialize N separate arrays with zeros
    mixtures = [np.zeros((num_channels, mixture_samples))
                for _ in range(len(audio_arrays))]

    for i, audio in enumerate(audio_arrays):
      channels, arr_samples = audio.shape

      if channels != num_channels:
        raise ValueError('All audio arrays must have the same number of '
                        'channels.')

      # Set pos to place the impulse
      current_pos = self.impulses[i].idx

      # Place the impulse in the corresponding mixture array
      mixtures[i][:, current_pos:current_pos + arr_samples] += audio

    self.impulse_mixture = mixtures

  def _final_reverb(self):
    """
    Apply the same reverb to the mixture, background and impulses
    """
    # random element of self.ir_file_paths
    random_ir_df = self.irs_pdf.sample(1)
    self.ir_file_path = random_ir_df['file_path'].values[0]

    # Load ir
    self.ir_sig, _ = utils.import_audio_file(
      self.ir_file_path, sr=self.sr, mono=True, return_offset=False)

    # Convolve with background getting only the first n_samples of the convolution
    n_channels = self.background_mixture.shape[0]
    n_samples = self.background_mixture.shape[1]
    
    # Convolve with impulse
    if self.n_impulses == 0:
      self.impulse_mixture = np.array([
        sig.convolve(self.impulse_mixture[ch], self.ir_sig[ch], mode='full',method="auto")[:n_samples]
        for ch in range(n_channels)
      ])
    else: 
      for i in range(len(self.impulse_mixture)):
        self.impulse_mixture[i] = np.array([
          sig.convolve(self.impulse_mixture[i][ch], self.ir_sig[ch], mode='full',method="auto")[:n_samples]
          for ch in range(n_channels)
        ])
    
    # Convolve with background
    self.background_mixture = np.array([
      sig.convolve(self.background_mixture[ch], self.ir_sig[ch], mode='full',method="auto")[:n_samples]
      for ch in range(n_channels)
    ])
    
    # Convolve with mixture
    self.mixture = np.array([
      sig.convolve(self.mixture[ch], self.ir_sig[ch], mode='full',method="auto")[:n_samples]
      for ch in range(n_channels)
    ])
    

  def generate_scene(self):
    """Create mixture with background(s) and impulses. After loading the
    backgrounds and impulses the following algorithm creates the mixture:

    1. Impulse placement on mixture: a random index is selected for each
    impulse in the mixture avoidint overlapping between impulses.
    
    2. Amplitude adjustment: The impulse's amplitude is adjusted to achieve
    the target SNR between the impulse and its proportion of background in
    the mixture.
    
    3. The mixture np.array (n_channels, n_samples) of the mixture is created.

    4. Audio data and metadata (json) are stored in output directory
    """
    # Load backgrounds FIRST 
    self._load_backgrounds()

    # Load impulses
    if self.n_impulses > 0:
      self._load_impulses()

      # Get each impule's index (posititon) in the mixture
      self._get_impulses_idx_on_mixture()

      # Adjust the SNR between the impulse and the proportion of the background
      # at the impulse's position
      self._adjust_impulses_amplitude_for_snr()

      # Create mixture array
      self._create_impulses_mixture()
      
    else:
      self.impulse_mixture = np.zeros(self.background_mixture.shape)

    # Create mixture (backgrounds + impulses)
    self.mixture = self.background_mixture + np.sum(self.impulse_mixture, axis=0)
    
    if config.FINAL_REVERB and self.irs_pdf:
      self._final_reverb()  
  

    # Normalize in full-scale ([-1, 1])
    self.mixture, self.normalization_gain = utils.normalize_signal(self.mixture)

    # Save audio
    self._save_audio()

    # Save metadata to json
    self._save_metadata()

  def _save_audio(self):
    """Save audio (impulses mixture, background mixture, scene mixture) to
    output dir.
    """

    output_dir = self.audio_output_dir
    
    
    impulse_sig = np.sum(self.impulse_mixture, axis=0).flatten()
    background_sig = self.background_mixture.flatten()
    background_sig = background_sig * self.normalization_gain

    # Apply gains
    impulse_sig = impulse_sig * self.normalization_gain
    mixture_sig = self.mixture.flatten()

    if self.scene_index is not None and isinstance(self.scene_index, int):
      impulse_filename = f'impulses_{self.scene_index}.wav'
      background_filename = f'background_{self.scene_index}.wav'
      mixture_filename = f'mixture_{self.scene_index}.wav'
    else:
      impulse_filename = 'impulses.wav'
      background_filename = 'background.wav'
      mixture_filename = 'mixture.wav'

    if self.scene_index is not None:
      self._extra_metadata = {
          'scene_index': self.scene_index,
          'impulse_audio_path': os.path.join(output_dir, impulse_filename),
          'background_audio_path': os.path.join(output_dir, background_filename),
          'mixture_audio_path': os.path.join(output_dir, mixture_filename)
        }
    else: 
      self._extra_metadata = {
          'impulse_audio_path': os.path.join(output_dir, impulse_filename),
          'background_audio_path': os.path.join(output_dir, background_filename),
          'mixture_audio_path': os.path.join(output_dir, mixture_filename)        
      }
    
    self.update_metadata(self._extra_metadata)

    sf.write(os.path.join(output_dir, impulse_filename), impulse_sig, self.sr)
    sf.write(os.path.join(output_dir, background_filename), background_sig, self.sr)
    sf.write(os.path.join(output_dir, mixture_filename), mixture_sig, self.sr)


  def _save_metadata(self):

    output_dir = self.json_output_dir
    
    if self.scene_index is not None and isinstance(self.scene_index, int):
      filename = f'metadata_{self.scene_index}.json'
    else:
      filename = 'metadata.json'
      
    with open(os.path.join(output_dir, filename),
              mode='w', encoding='utf-8') as json_file:
      json.dump(self.metadata, json_file, indent=4)


  def plot_mixture(self):
    """
    Plot the waveform of self.impulse_mixture with each impulse (defined by
    start_idx and end_idx from self.metadata) colored differently.
    Additionally, plot the background signal.
    """
    if not hasattr(self, 'metadata') or not self.metadata:
      raise ValueError('Metadata is missing. Ensure impulses have been '
                       'tracked properly.')

    if self.n_impulses > 0:
      if not hasattr(self, 'impulse_mixture') or self.impulse_mixture is None:
        raise ValueError('The mixture has not been created yet. '
                         'Call create_impulses_mixture() first.')
      else:
        # Extract the mixture and its shape
        impulses_mixture = np.sum(self.impulse_mixture, axis=0)
        num_channels = impulses_mixture.shape[0]
    else:
      num_channels = 1

    num_samples = self.background_mixture.shape[1]

    # Prepare the time axis
    time = np.linspace(0, num_samples / self.sr, num_samples)

    # Define a color palette for the impulses
    colors = plt.cm.tab10.colors  # Use a predefined color map

    # Create the plot
    plt.figure(figsize=(15, 5))

    for ch in range(num_channels):
      plt.subplot(num_channels, 1, ch + 1)
      plt.title(f'Channel {ch + 1} - Colored Impulse Contributions')
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')

      # Plot the background signal
      plt.plot(time,
               self.background_mixture[ch],
               alpha=0.3,
               color='black',
               linestyle='--',
               label=f'Background ({self.background_scene})'
              )

      if self.n_impulses > 0:
        # Color each impulse
        for i, impulse in enumerate(self.impulses):
          start_idx = impulse.idx
          end_idx = start_idx + impulse.data.shape[1]

          # Time range for the impulse
          impulse_time = time[start_idx:end_idx]

          # Plot the impulse contribution with a unique color
          plt.plot(impulse_time,
                  impulses_mixture[ch, start_idx:end_idx],
                  color=colors[i % len(colors)],
                  linewidth=2,
                  label=f'Impulse {i} | SNR = {impulse.snr} | label = {impulse.std_label} '
                  )

      plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



class Component():
  """Parent class for backgrounds/impulses
  """

  @property
  def offset(self):
    return self._offset

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, value):
    self._data = value

  def __init__(self, std_label, file_path, sr, duration, mono, offset):
    assert os.path.isfile(file_path), f'{file_path} is not a file!'

    self.file_path = file_path
    self.std_label = std_label
    self.sr = sr
    self.duration = duration
    self.mono = mono

    # Audio import metadata
    self._offset = offset
    self._data = None

    # Salt mapping
    self.dataset_labels = None

    self._load_data()

  def _load_data(self):
    """Load audio data, sample rate and offset of the component
    """
    # Import audio file
    self._data, _, self._offset = utils.import_audio_file(
      file_path=self.file_path,
      sr=self.sr,
      duration=self.duration,
      mono=self.mono,
      offset=self.offset,
      return_offset=True,
      print_warning=True
    )

    # Set duration
    self.duration = self._data.shape[1] / self.sr

  def _init_data_augmentor(self):
    pass

  def apply_data_augmentation(self):
    pass



class Impulse(Component):
  """Class to handle a single Impulse sound event
  """
  @property
  def metadata(self):
    return {
      'file_path': self.file_path,
      'sr': self.sr,
      'duration': self.duration,
      'n_channels': 1 if self.mono else self._data[0],
      'mono': self.mono,
      'offset': self.offset,
      'snr': self.snr,
      'idx': self.idx,
      'std_label': self.std_label,
      'augmentation_params': self._get_augmentation_params_to_dict()
    }

  def _get_augmentation_params_to_dict(self):
    """Function to make the augmentation parameters (function names,
    arguments, etc.) a dictionary of strings and numbers.

    Returns
    -------
    dict
        Dict with strings of function names and parameters for the augmentation
    """
    return {
      obj.__class__.__name__: {
        key: value for key, value in obj.__dict__.items() if isinstance(value, (int, float, str))
      }
      for obj in self.augmentation_params
    }

  def __init__(self,
               std_label : str,
               file_path : str,
               sr : int,
               duration : float,
               mono : bool,
               impulse_snr : float,
               augment : bool=False,
               ir_file_paths = None,):
    """Impulse instance

    Parameters
    ----------
    std_label : str
        the impule's standard label defined in SALT
    file_path : str
        file path to the audio file
    sr : int
        sample rate
    duration : float
        impulse duration
    mono : bool
        True if mono, else False
    impulse_snr : float
        the snr between the impulse and the background in dB.
        Positive values -> louder impulse
        Negative values -> louder background
    """
    super().__init__(std_label, file_path, sr, duration, mono, offset=0)

    self.augment = None
    self.ir_file_paths = ir_file_paths
    self.snr = impulse_snr
    self.augmentation_params = config.IMPULSE_AUGMENTATION_PARAMS
    self.idx = None # Impulse position in the mixture. Defined in Scene class

    if augment:
      self._init_data_augmentor()
      self.apply_data_augmentation()

  def _init_data_augmentor(self):
    
    if self.ir_file_paths is not None:
      augmentation_params = self.augmentation_params + \
        [ApplyImpulseResponse(ir_path=self.ir_file_paths, p=0.9, leave_length_unchanged=False)]
    else:
      augmentation_params = self.augmentation_params
    
    self.augment = Compose(augmentation_params)

  def apply_data_augmentation(self):
    self.augment(self._data, sample_rate=self.sr)

  def remove_edge_silence(self, threshold=1e-7):
    # Remove edge silence
    self._data = utils.remove_signal_edge_silence(self._data, threshold)

    # Update impulse duration after silence removal
    self.duration = self._data.shape[1] / self.sr



class Background(Component):
  """Class to handle a single Impulse sound event
  """
  @property
  def metadata(self):
    return {
      'file_path': self.file_path,
      'sr': self.sr,
      'duration': self.duration,
      'mono': self.mono,
      'offset': self.offset,
      'std_label': self.std_label,
    }

  def __init__(self,
               std_label : str,
               file_path : str,
               sr : int,
               duration : float,
               mono : bool,
               ir_file_paths = None, 
               augment : bool=False):
    """Background instance

    Parameters
    ----------
    std_label : str
        the scene label as defined in SALT
    file_path : str
        the path to the audio file
    sr : int
        sample rate
    duration : float
        the audio file's duration in secs
    mono : bool
        True if mono, else False
    irs_pdf : pd.DataFrame
        pdf metadata of Impulse Responses
    """
    super().__init__(std_label, file_path, sr, duration, mono, offset=-1)

#    self.ir_file_path = None
    self.rt_60 = None
    self.ir_type = None
    self.ir_sig = None
    self.all_ir_file_paths = ir_file_paths

    if augment:
      self.apply_data_augmentation()
      
  def get_ir_file_paths(self):
    return self.irs_pdf['file_path'].values.tolist()

  def apply_data_augmentation(self):

    augment = Compose([ApplyImpulseResponse(ir_path=self.all_ir_file_paths, p=0.5, leave_length_unchanged=True)])
    self._data = augment(self._data, sample_rate=self.sr)
    

