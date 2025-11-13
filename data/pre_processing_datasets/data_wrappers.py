"""
Abstract base class DatasetManager for all corpus database manager/loader

:Authors:
    Paraskevas Stamatiadis <stamatiadis@telecom-paris.fr>
"""
from abc import ABCMeta, abstractmethod  # abstract classes
from tqdm import tqdm
from ast import literal_eval

import os
import re
import sys
import copy
import json
import subprocess
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

class DatasetManager(object):
  """
  Dataset manager class
  """
  __metaclass__ = ABCMeta

  def __init__(self, metadata_file_path, load_metadata=True):
    self._metadata_file_path = metadata_file_path
    self._pdf_metadata = None

    # Pandas dataframe stored in metadata_file_path
    if os.path.isfile(self._metadata_file_path) and load_metadata:
      self.load_metadata_file()

  @property
  def pdf_metadata(self):
    return self._pdf_metadata

  @property
  def root_dir(self):
    return self._root_dir

  @property
  def file_paths(self):
    return self._file_paths

  @property
  def metadata_file_path(self):
    return self._metadata_file_path

  @property
  def info(self):
    return self._info

  @abstractmethod
  def generate_metadata_file(self):
    pass

  def load_metadata_file(self):
    """
    Load the metadata file specified in self._metadata_file_path into a
    panda dataframe objet (self._pdf_metadata) and replace relative paths
    to audio files by absolute paths
    """

    if not os.path.exists(self._metadata_file_path):
      self.generate_metadata_file()
    else:
      print(type(self).__name__ + ': load metadata file ' +
            self._metadata_file_path)

    # -- Load metadata file
    self._pdf_metadata = pd.read_csv(self._metadata_file_path,
                                     index_col=0,
                                     keep_default_na=False,
                                     low_memory=False)

    # -- Replace relative path by absolute path
    root_dir = os.path.dirname(self._metadata_file_path)
    for col in self.get_pdf_columns_with_filepaths(self._pdf_metadata):
      abs_paths = [os.path.abspath(os.path.join(root_dir, rel_path))
                   for rel_path in self._pdf_metadata[col].tolist()]      
      self._pdf_metadata[col] = abs_paths
    return

  def reset_metadata_file(self):
    """
    Returns a bool to whether or not regenerate the metadata file
    specified in self.metadata_file_path
    """

    print(type(self).__name__ + ': generate metadata file')

    overwrite = self.confirm_overwriting(self._metadata_file_path,
                                         exit_python=False)

    if overwrite:
      self._pdf_metadata = None
    else:
      print('Loading ' + self._metadata_file_path + ' instead')
      self.load_metadata_file()
    return overwrite

  @staticmethod
  def confirm_overwriting(file_path, exit_python=True):

    """
    Check if a file/directory exists and open prompt to ask for a manual
    confirmation for overwriting.

    Parameters
    ----------
    file_path: str
      file or directory path

    exit_python: bool (default True)
      whether or not exiting python if the answer is to not overwrite
      the existing file/directory

    Returns
    -------
      create_new_file: bool
        in case exit_python=False
    """
    create_new_file = True
    prompt_prefix = '!' * 5 + ' '

    if os.path.isfile(file_path) or os.path.isdir(file_path):

      print(prompt_prefix + file_path + ' already exists')

      key_input = ''
      while key_input.lower() != 'y' and key_input.lower() != 'n':
        key_input = input(prompt_prefix + 'OVERWRITE ANYWAY? [y/n]')

      if key_input.lower() == 'n':
        create_new_file = False

    if exit_python and not create_new_file:
      print('exiting')
      sys.exit()
    else:
      return create_new_file


  def get_database(self, filters=None, dataframe=True):
    """
    Get database metadata as a pandas dataframe object or a list of
    dictionary. Filter the content of the database.

    Parameters
    ----------
    filters: dict or None (default None)
      pandas filters stored in a dictionary
      {column_name: allowed_values}

    dataframe: bool (default True)
      if True returns the database as a pandas dataframe object,
      else returns a list of dictionary (each element of the list
      correspond to a recording session)
    """

    self.load_metadata_file()  # reload to get absolute path of files

    pdf_metadata = self._pdf_metadata.copy(deep=True)  # cp before filt

    if filters is not None:
      for key in list(filters):
        pdf_metadata = self.filter_pdf(pdf_metadata, key, filters[key])

    pdf_metadata.reset_index(drop=True, inplace=True)

    if dataframe:
      return pdf_metadata
    else:
      return pdf_metadata.to_dict('records')

  def get_total_duration(self):
    """
    Returns total duration (in sec) of all samples composing the database
    """

    if self._pdf_metadata is None:
      self.load_metadata_file()

    return self._pdf_metadata['duration'].sum()

  def get_abs_path(self, file_rel_path):
    root_dir = os.path.dirname(self._metadata_file_path)
    if file_rel_path.startswith('./'):
      file_rel_path = file_rel_path[2:]
    if file_rel_path.startswith('.'):
      file_rel_path = file_rel_path[1:]
    return os.path.join(root_dir, file_rel_path)

  def get_pdf_columns_with_filepaths(self, pd_dataframe):
    """
    Parse pandas DataFrame to retrieve columns corresponding to existing
    file/directory paths

    Parameters
    ----------
    pd_dataframe: pd.DataFrame instance

    Returns
    -------
    list of str of column names

    """
    column_names = []

    for col in pd_dataframe.columns.to_list():
      # test first element (index = 0)
      elem = pd_dataframe[col][0]
      if isinstance(elem, str):
        if os.path.isfile(elem) or \
           os.path.isfile(self.get_abs_path(elem)):
          column_names.append(col)

    return column_names


  def create_csv_metadata_file(self):
    """Creates file_paths.csv file containing the relative paths for all
     the audio files of the dataset

    Parameters
    ----------
    root_dir : string
        The root directory of the dataset

    Raises
    ------
    ValueError
        raises the error when the root directory does not exist
    """
    root_dir = os.path.dirname(self._metadata_file_path)
    filepaths_path = os.path.join(root_dir, 'file_paths.csv')

    if os.path.isfile(filepaths_path):
      return
    # else create file_paths.csv file

    cwd = os.getcwd()
    try:
      os.chdir(root_dir)
    except FileNotFoundError as exc:
      raise ValueError('Directory does not exist') from exc
    else:
      command = """find * -name "*.flac" -o -name "*.wav" -o -name "*.ogg"\
| awk -F'\n' -v OFS='\t' '{print $0}' > file_paths.csv"""
      subprocess.run(command, shell=True, check=False)
    finally:
      os.chdir(cwd)

    return

  def load_audio_filepaths(self, id_column_name='file_id', rm_extention=True):
    """Loads the file_paths.csv file in a pandas DataFrame

    Parameters
    ----------
    id_column_name : str, optional
        id column regarding on the type of the file and the dataset.
        e.g. "file_id" for MAVD-traffic, "video_id" for AudioSet etc.
        , by default 'file_id'
    rm_extention : bool, optional
        If true the extention of the file's path will be removed
        , by default True

    Returns
    -------
    pandas DataFrame
        DataFrame with the relative paths of all audio files in sub-
        directories of the dataset: {file_path, file_id}
    """

    # - Set root_dir and file_paths.csv path
    root_dir = os.path.dirname(self._metadata_file_path)
    filepaths_path = os.path.join(root_dir, 'file_paths.csv')

    # - If file_paths.csv does not exist, create it
    if not os.path.isfile(filepaths_path):
      self.create_csv_metadata_file()

    filepaths_df = pd.read_csv(filepaths_path, names=['file_path'], sep='\t')

    if len(filepaths_df.index) != 0:
      # - Create the id column. Id can differ depending on dataset.
      filepaths_df[id_column_name] = filepaths_df['file_path'].apply(os.path.basename)

      # - Remove extention if rm_extention is True
      if rm_extention:
        filepaths_df[id_column_name] =\
          filepaths_df[id_column_name].str.rsplit(pat='.', n=1, expand=True)[0]
    else:
      filepaths_df[id_column_name] = ''

    return filepaths_df

  @staticmethod
  def filter_pdf(pd_dataframe, column_name, column_values):
    """
    Filter a pandas DataFrame

    Parameters
    ----------
    pd_dataframe: pd.DataFrame instance

    column_name: str
      column along which the filtering is applied

    column_values: list
      list of values allowed for which dataframe indices are kept

    Returns
    -------
    the filtered dataframe
    """

    if not isinstance(column_values, list):
      column_values = [column_values]

    return pd_dataframe.loc[pd_dataframe[column_name].isin(column_values)]

  def get_audio_file_metadata(self, file_path,
                              get_cutoff=False,
                              raise_error=False):
    """
    Get metadata (sampling_rate, n_channels, duration and cutoff
    frequency) of an audio file

    Parameters
    ----------
    file_path: str
      path to an audio file

    get_cutoff: bool
      True to get the cutoff frequency

    raise_error: bool (default)
      raise an error if reading the metadata from the audio file failed

    Returns
    -------
    metadata dictionary
    """

    audio_md = None
    data = None
    sample_rate = None
    cutoff_freq = None

    abs_file_path = self.get_abs_path(file_path)

    # --- Read audio file
    try:
      data, sample_rate = sf.read(abs_file_path)
    except (TypeError, RuntimeError, ValueError, MemoryError) as err:
      if raise_error:
        raise err
      else:
        return {'n_channels': '',
                'sampling_rate': '',
                'duration': '',
                'cutoff_freq': ''}

    # If corrupted file
    if data is None:
      return {'n_channels': '',
              'sampling_rate': '',
              'duration': '',
              'cutoff_freq': ''}
    # else calculate audio metadata

    # - duration in secs
    duration = len(data) / sample_rate
    data = data.T

    # - Compute num of channels
    n_channels = 1 if data.ndim == 1 else data.shape[0]

    # --- Compute cutoff frequency of audio file
    if get_cutoff:
      # - FFT parameters
      n_fft = 2048
      hop_size = 512

      # - Initializations
      avg_density = np.zeros((int(1+n_fft/2)))
      freq_axis = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

      # - Convert to Mono
      if n_channels > 1:
        data = librosa.to_mono(data)

      abs_stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_size))
      sum_freq = np.sum(abs_stft, axis=1)

      # - Compute average spectral density
      if np.max(sum_freq) != 0:
        avg_density += sum_freq/np.max(sum_freq)

      # - Choose the cutoff frequency as the frequency which explains
      # the 95% of the energy.
      total_energy = np.sum(avg_density)
      j=0
      while np.sum(avg_density[:j]) / total_energy < 0.95:
        j+=1

      if j >= len(freq_axis):
        j=len(freq_axis)-1

      cutoff_freq = int(freq_axis[j])

    # - Create audio metadata dictionary
    if get_cutoff:
      audio_md = {'n_channels': n_channels,
                  'sampling_rate': sample_rate,
                  'duration': duration,
                  'cutoff_freq': cutoff_freq}
    else:
      audio_md = {'n_channels': n_channels,
                  'sampling_rate': sample_rate,
                  'duration': duration}
    return audio_md

  def read_audio_file_metadata(self, file_path):
    """
    Read metadata: sampling_rate, n_channels, duration and cutoff
    frequency (if computed) of an audio file from self._pdf_metadata

    Parameters
    ----------
    file_path: str
      path to an audio file

    Returns
    -------
    metadata dictionary
    """
    audio_md = {}

    if self._pdf_metadata is None:
      self.load_metadata_file()

    # - Convert relative path to absolute
    abs_path = self.get_abs_path(file_path)

    audio_metadata_list = [
      'sampling_rate',
      'n_channels',
      'duration',
      'cutoff_freq'
    ]

    # - Get dataframe record using file_path
    metadata_df = self._pdf_metadata.loc[
      self._pdf_metadata['file_path'] == abs_path]

    # - Remove extra columns (usually due to strong annotations)
    metadata_df = metadata_df.drop_duplicates(subset='file_path')

    metadata_columns = metadata_df.columns

    for audio_metadata in audio_metadata_list:
      if audio_metadata in metadata_columns:
        audio_md[audio_metadata] = metadata_df[audio_metadata].values[0]

    return audio_md

  def _add_audio_metadata_to_pdf_metadata(
      self,
      pdf_metadata,
      file_path_column_name='file_path',
      get_cutoff=False,
    ):
    """Adds audio specific metadata to pdf_metadata

    Parameters
    ----------
    pdf_metadata : pandas DataFrame
        Dataset's DataFrame with metadata

    file_path_column_name : string
        name of the column used for file paths, by default 'file_path'

    use_multiprocessing : bool
        If True will use multi_processing, by default False

    Returns
    -------
    pandas DataFrame
        pdf_metadata with the audio spesific metadata included
    """
    # -- Get filepaths list
    filepaths_list = pdf_metadata[file_path_column_name].unique()
    # -- Compute sample_rate, n_channels and cutoff_freq for all files
    audio_md_list = []
    for rel_path in tqdm(filepaths_list, desc='Generating audio metadata'):
      # - Get absolute path of file
      abs_path = self.get_abs_path(os.path.join(self._root_dir, rel_path))
      # - Get audio file's metadata
      audio_file_md = self.get_audio_file_metadata(abs_path, get_cutoff)

      # - Set file_path
      audio_file_md[file_path_column_name] = rel_path

      # - Append to audio metadata list
      audio_md_list.append(audio_file_md)

    # --- Finally, merge with the existing metadata
    audio_metadata_df = pd.DataFrame(audio_md_list)

    pdf_metadata = pd.merge(
      pdf_metadata,
      audio_metadata_df,
      how='left',
      on=file_path_column_name)

    return pdf_metadata


########################################################################
############################ ESC-50 Wrapper ############################
########################################################################

class ESC50(DatasetManager):
  """
  Subclass of EventCorpus for manipulating the ESC-50 dataset for sound 
  event detection
  """

  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # --- Load file paths
    file_paths = self.load_audio_filepaths(rm_extention=False)

    pdf_metadata = pd.read_csv(
        os.path.join(self._root_dir, 'meta', 'esc50.csv'),
        skiprows=1,
        names=['file_id', 'fold', 'target', 'label', 'esc10', 'src_file', 'take']
    )

    # --- Add file path info
    pdf_metadata = pd.merge(
        pdf_metadata, file_paths, how='left', on='file_id')

    # --- Add audio-specific metadata
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)

    # --- Save metadata file
    pdf_metadata.to_csv(self._metadata_file_path)

########################################################################
########################### ReaLISED Wrapper ###########################
########################################################################

class ReaLISED(DatasetManager):
  """
  Subclass of EventCorpus for manipulating the ReaLISED dataset for
  sound event detection.
  """

  _label_map = {
      'bea': 'beater',
      'coo': 'cooking',
      'cup': 'cupboard/wardrobe',
      'dis': 'dishwasher',
      'dra': 'drawer',
      'fur': 'furniture movement',
      'mic': 'microwave',
      'obj': 'object falling',
      'smo': 'smoke extractor',
      'spe': 'speech',
      'swi': 'switch',
      'tel': 'television',
      'vac': 'vacuum cleaner',
      'wal': 'walking',
      'was': 'washing machine',
      'wat': 'water tap',
      'win': 'window'
  }

  _actions_map = {
      '01': 'close',
      '02': 'open',
      '03': 'throw',
      '04': 'turn on',
      '05': 'turn off',
      '06': 'move',
      '07': 'plug',
      '08': 'unplug',
      '09': 'raise',
      '10': 'lower',
      '00': 'no info'
  }

  _material_map = {
      '01': 'wood',
      '02': 'glass',
      '03': 'metal',
      '04': 'plastic',
      '05': 'ceramic',
      '06': 'synthetic',
      '07': 'cardboard',
      '08': 'marble',
      '09': 'floating platform',
      '10': 'platelet',
      '11': 'wicket',
      '12': 'carpet',
      '13': 'medium-density fibreboard MDF',
      '00': 'no info'
  }

  _intensity_map = {
      '1': 'low',
      '2': 'medium',
      '3': 'high',
      '0': 'no info'
  }

  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def load_metadata_file(self):
    super().load_metadata_file()
    self._pdf_metadata = self._pdf_metadata.loc[
      self._pdf_metadata['label'] != '']

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file
    file_paths = self.load_audio_filepaths()

    # Extract metadata from filenames
    pdf_metadata = file_paths.copy(deep=True)
    pdf_metadata[['label', 'n_event', 'action', 'material', 'intensity']
                 ] = pdf_metadata['file_id'].str.split('_', expand=True)

    # Get label from ids
    pdf_metadata['label'] = pdf_metadata['label'].map(self._label_map)
    pdf_metadata['action'] = pdf_metadata['action'].map(self._actions_map)
    pdf_metadata['material'] = pdf_metadata['material'].map(self._material_map)
    pdf_metadata['intensity'] = pdf_metadata['intensity'].map(
        self._intensity_map)

    # Add audio-specific metadata and save to csv
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
########################## VocalSound Wrapper ##########################
########################################################################

class VocalSound(DatasetManager):
  """Subclass of EventCorpus for manipulating the VocalSound dataset
  for human non-speech sound detection.
  https://github.com/YuanGongND/vocalsound
  """
  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # Read metadata for speakers per split
    meta_dir = os.path.join(self._root_dir, 'meta')
    pdf_metadata = pd.DataFrame()
    for split in ['tr', 'te', 'val']:
      split_df = pd.read_csv(
        filepath_or_buffer=os.path.join(meta_dir, f'{split}_meta.csv'),
        names=['spk_id', 'gender', 'age', 'country', 'language', 'health_condition'])

      if split == 'tr':
        split_df['set'] = 'train'
      elif split == 'val':
        split_df['set'] = 'val'
      else:
        split_df['set'] = 'test'

      pdf_metadata = pd.concat([pdf_metadata, split_df], ignore_index=True)

    # Read audio files
    filenames = os.listdir(os.path.join(self._root_dir, 'data_44k'))
    file_paths = [os.path.join('data_44k', f) for f in filenames]

    spk_ids = [f.rsplit('.', 1)[0].rsplit('_')[0] for f in filenames]
    labels = [f.rsplit('.', 1)[0].rsplit('_')[2] for f in filenames]

    # Merge with pdf_metadata
    pdf_metadata = pd.merge(
      pd.DataFrame({'spk_id': spk_ids,
                    'file_path': file_paths,
                    'label': labels}),
      pdf_metadata,
      how='left',
      on='spk_id'
    )

    # Add audio-related metadata
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)

    # Save to csv
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
########################## Nonspeech7k Wrapper #########################
########################################################################

class Nonspeech7k(DatasetManager):
  """
  Subclass of EventCorpus for manipulating the Nonspeech7k dataset
  for sound event detection.
  https://zenodo.org/records/6967442
  """

  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    train_md_df = pd.read_csv(os.path.join(self._root_dir, 'metadata of train set .csv'))
    test_md_df = pd.read_csv(os.path.join(self._root_dir, 'metadata of test set.csv'))    

    for df in [train_md_df, test_md_df]:
      df.rename(columns={
				'Filename': 'filename',
				'File ID': 'file_id',
				'Duration in ms': 'duration_ms',
				'Class ID': 'label_id',
				'Classname': 'label',
				'augmentation  id': 'augmentation_id',
				'Augmentation  type': 'augmentation_type'
		}, inplace=True)

    if "only_impulses" in self.root_dir:
      # remove the non impulsive audios from metadata (the one that were not copied in the new dataset)      
      
      # get all the filename ending with .wav in self.root_dir/train
      train_audio_files = os.listdir(os.path.join(self._root_dir, 'train'))
      train_audio_files = [f for f in train_audio_files if f.endswith('.wav')]
      train_md_df = train_md_df[train_md_df['filename'].isin(train_audio_files)]   
 
      test_audio_files = os.listdir(os.path.join(self._root_dir, 'test'))
      test_audio_files = [f for f in test_audio_files if f.endswith('.wav')]
      test_md_df = test_md_df[test_md_df['filename'].isin(test_audio_files)]

    # Add file path
    train_md_df['file_path'] = train_md_df['filename'].apply(
      lambda x: os.path.join('train', x))

    test_md_df['file_path'] = test_md_df['filename'].apply(
      lambda x: os.path.join('test', x))

    # Add split
    train_md_df['set'] = 'train'
    test_md_df['set'] = 'test'

    pdf_metadata = pd.concat([train_md_df, test_md_df], ignore_index=True)

    # Re-arrange for better readability
    pdf_metadata = pdf_metadata[[
        'file_path', 'filename', 'file_id', 'label_id', 'label', 'set',
        'augmentation_id', 'augmentation_type', 'source', 'duration_ms'
    ]]

    # Add audio-specific metadata and save to csv
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)
    pdf_metadata.to_csv(self._metadata_file_path)

########################################################################
################## FreesoundOneShotPercussive Wrapper #################
########################################################################

class FreesoundOneShotPercussive(DatasetManager):
	"""
	Subclass of MusicCorpus to manipulate the Freesound One-Shot Percussive
	Sounds dataset
	https://zenodo.org/records/4687854
	"""
	def __init__(self, root_dir):
		assert os.path.isdir(root_dir)
		self._root_dir = root_dir
		md_file_path = os.path.join(self._root_dir, 'metadata.csv')
		super().__init__(metadata_file_path=md_file_path)

	def generate_metadata_file(self):
		if not self.reset_metadata_file():
			return
		# else generate metadata file

		# - List all wav and json files
		wav_file_paths = []
		annotation_paths = []
		for root, _, files in os.walk(os.path.join(self._root_dir)):
			for file in files:
				if file.endswith('.wav'):
					wav_file_paths.append(os.path.join(root, file))
				elif file.endswith('.json') and file != 'sound_info_analysis.json':
					annotation_paths.append(os.path.join(root, file))
				else:
					pass

		# --- Read metadata for each file
		pdf_metadata = pd.DataFrame()
		for annot_fp in tqdm(annotation_paths, desc='Reading analysis metadata'):
			try:
				with open(annot_fp, encoding='utf-8') as f:
					data = json.load(f)

					# Add file id as index for the dataframe
					data['id'] = [annot_fp.rsplit('/', 1)[1].split('_')[0]]
			except ValueError as e:
				print(annot_fp)
				raise e

			pdf_metadata = pd.concat([pdf_metadata, pd.DataFrame(data)],
															 ignore_index=True)

		# - Rename cols for consistency across all loaders
		pdf_metadata.rename(columns={'channels': 'n_channels',
																 'samplerate': 'sampling_rate'}, inplace=True)

		# - Add file paths info
		ids = [int(f.rsplit('/', 1)[1].rsplit('.')[0]) for f in wav_file_paths]
		fps_df = pd.DataFrame({'file_path': wav_file_paths, 'id': ids})

		# Ensure 'id' in both DataFrames are of the same type
		fps_df['id'] = fps_df['id'].astype(str)
		pdf_metadata['id'] = pdf_metadata['id'].astype(str)

		pdf_metadata = pd.merge(fps_df, pdf_metadata, how='left', on='id')

		# --- Read sound info analysis metadata
		analysis_pdf = pd.read_json(os.path.join(self._root_dir,
																						 'sound_info_analysis.json'))

		# - Drop duration col (exists in the rest of metadata)
		analysis_pdf.drop(columns=['duration'], inplace=True)
		analysis_pdf['id'] = analysis_pdf['id'].astype(str)

		pdf_metadata = pd.merge(pdf_metadata, analysis_pdf, how='left', on='id')
		pdf_metadata = pdf_metadata.reset_index(drop=True)

		# - Abs to rel path
		pdf_metadata['file_path'] = pdf_metadata['file_path'].apply(
			lambda file_path: os.path.relpath(file_path, self._root_dir))

		# --- Save to csv
		pdf_metadata.to_csv(self._metadata_file_path)

	def filter_pdf_with_tags(self, tags):
		df = self._pdf_metadata.copy(deep=True)
		df['tags'] = df['tags'].apply(literal_eval)
		df = df.explode(column=['tags'])
		return df.loc[df['tags'].isin(tags)]


########################################################################
############################# ARTE Wrapper ############################
########################################################################

class Arte(DatasetManager):
  """Subclass of SceneCorpus for manipulating the ARTE dataset for ASC.
  """
  _eq_map_ = {
    'noEQ': False,
    'withEQ': True
  }

  _label_map_ = {
    'Library': 'library',
    'Office': 'office',
    'Church': 'church',
    'Living_Room': 'living_room',
    'Church_1': 'church',
    'Church_2': 'church',
    'Cafe_1': 'cafe',
    'Cafe_2': 'cafe',
    'Dinner_party': 'dinner_party',
    'Street_Balcony': 'street_balcony',
    'Train_Station': 'train_station',
    'Food_Court_1': 'food_court',
    'Food_Court_2': 'food_court'
  }

  def __init__(self, root_dir=None):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # List audio files
    file_paths = []
    for root, _, files in os.walk(self._root_dir):
      for file in files:
        file_ext = file.rsplit('.', 1)[1]
        if file_ext in ['wav', 'flac', 'mp3', 'ogg']:
          if 'binaural' in file:
            if not ('RIR' in file or 'Diffuse_noise' in file):
              file_paths.append(os.path.join(root, file))

    # Get relative paths
    file_paths = [f.split(self._root_dir)[1][1:] for f in file_paths]

    pdf_metadata = pd.DataFrame({'file_path': sorted(file_paths)})

    # Add label
    pdf_metadata['label'] = pdf_metadata['file_path'].apply(
      lambda x: os.path.basename(x).split('_', 1)[1].rsplit('_', 2)[0])

    pdf_metadata['label'] = pdf_metadata['label'].map(self._label_map_)

    # Add eq_applied col
    pdf_metadata['eq_applied'] = pdf_metadata['file_path'].apply(
      lambda x: x.rsplit('.', 1)[0].rsplit('_', 1)[1])

    pdf_metadata['eq_applied'] = pdf_metadata['eq_applied'].map(self._eq_map_)

    # Add audio specific metadata
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)

    # Save to csv
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
########################### CAS 2023 Wrapper ###########################
########################################################################

class Cas2023(DatasetManager):
  """Subclass of SceneCorpus for manipulating the CAS 2023 dataset for
  Acoustic Scene Classification
  https://zenodo.org/records/10616533
  """
  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # - Read provided metadata
    pdf_metadata = pd.read_csv(os.path.join(self._root_dir, 'ICME2024_ASC_dev_label.csv'))

    # - Replace Nan values (unspecified classes) with empty strings
    pdf_metadata.fillna('', inplace=True)

    # - Rename label column
    pdf_metadata.rename(columns={'scene_label': 'label', 'filename': 'file_id'}, inplace=True)

    # - Load audio file paths
    file_paths = self.load_audio_filepaths()

    # - Add audio file paths to pdf metadata
    pdf_metadata = pd.merge(file_paths, pdf_metadata, how='left', on='file_id')

    # Add audio-specific metadata and write to csv
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
########################## CochlScene Wrapper ##########################
########################################################################

class CochlScene(DatasetManager):
  """Subclass of SceneCorpus for manipulating the CochlScene dataset
  """
  def __init__(self, root_dir):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    data = []
    for root, _, files in os.walk(self._root_dir):
      for file in files:
        if file.endswith('wav'):
          abs_file_path = os.path.join(root, file)
          _, split, label, _ = abs_file_path.rsplit('/', 3)
          data.append({
             'file_path': os.path.relpath(abs_file_path),
             'split': split,
             'label': label
          })
    pdf_metadata = pd.DataFrame(data)

    # Add audio-specific metadata and save to csv
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
########################## DCASE 2018 Wrapper ##########################
########################################################################

class Dcase2018(DatasetManager):
  """
  Subclass of SceneCorpus for manipulating the DCASE 2018 dataset for
  acoustic scene classification Task 1
  http://dcase.community/challenge2018/task-acoustic-scene-classification
  """
  def __init__(self, root_dir=None):
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # --- Get metadata for all files
    metadata_list = []

    # --- Import Dcase provided metadata
    dcase_md_dic = {}

    # Get train/test split
    dset_annot_dir = os.path.join(self._root_dir, 'evaluation_setup')
    with open(
      file=os.path.join(dset_annot_dir, 'fold1_train.txt'),
      mode='r',
      encoding='utf-8') as f:

      train_list = [l.split('\t')[0] for l in f.read().splitlines()]

    with open(
      file=os.path.join(dset_annot_dir, 'fold1_test.txt'),
      mode='r',
      encoding='utf-8') as f:

      test_list = [l.split('\t')[0] for l in f.read().splitlines()]

    # Get list of files + metadata
    with open(
      file=os.path.join(self._root_dir,'meta.csv'),
      mode='r',
      encoding='utf-8') as f:

      md_list = f.read().splitlines()[1:]  # remove header

    for md_line in md_list:
      # [file_relative_path, label, session_id]
      md = md_line.split('\t')[:3]
      dcase_md_dic[md[0]] = {'label': md[1], 'session_id': md[2]}
      if md[0] in train_list:
        dcase_md_dic[md[0]]['set'] = 'train'
      elif md[0] in test_list:
        dcase_md_dic[md[0]]['set'] = 'test'
      else:
        raise ValueError("Cannot find 'set' metdata for the file "
                 + md[0])

    # - Loop over samples
    for file_rpath in tqdm(sorted(list(dcase_md_dic)), 'Generating metadata file'):

      file_path = os.path.normpath(
        os.path.join(self._root_dir, file_rpath))

      # - Get audio metadata
      md_dic = self.get_audio_file_metadata(file_path)
      if md_dic is None:  # file is not audio, iterate
        continue

      # - Set audio file path relative to metadata file
      md_dic['file_path'] = os.path.relpath(file_path, self._root_dir)

      # - Add Dcase provided metadata
      md_dic.update(copy.deepcopy(dcase_md_dic[file_rpath]))

      # - Validate and append
      metadata_list.append(md_dic)

    # --- Create panda data frame and export csv
    pdf_metadata = pd.DataFrame(metadata_list)
    pdf_metadata.to_csv(self._metadata_file_path)


########################################################################
###################### Vehicle Interior Wrapper ########################
########################################################################

class VehicleInteriorSound(DatasetManager):
  """Subclass of SceneCorpus for manipulating the Vehicle Interior Sound 
  dataset for acoustic scene classification.
  https://zenodo.org/records/5606504
  """
  _class_map = {
    1: 'Bus',
    2: 'Minibus',
    3: 'Pickup',
    4: 'Sports car',
    5: 'Jeep',
    6: 'Truck',
    7: 'Crossover',
    8: 'Car (C Class - 4K)'
  }

  def __init__(self, root_dir=None):
    assert os.path.isdir(root_dir)
    self._root_dir = root_dir
    md_file_path = os.path.join(self._root_dir, 'metadata.csv')
    super().__init__(metadata_file_path=md_file_path)

  def generate_metadata_file(self):
    if not self.reset_metadata_file():
      return
    # else generate metadata file

    # Get filenames, file paths and label ids from filenames
    filenames = os.listdir(os.path.join(self._root_dir, 'VISC Dataset SON'))
    file_paths = [os.path.join(self._root_dir, 'VISC Dataset SON', f) for f in filenames]
    label_ids = [int(f[0]) for f in filenames]

    pdf_metadata = pd.DataFrame({'file_path': file_paths,
                                 'filename': filenames,
                                 'label_id': label_ids})

    # Abs to rel paths
    pdf_metadata['file_path'] = pdf_metadata['file_path'].apply(
      lambda x: x.rsplit(self._root_dir)[1][1:])

    # Extract labels from label ids
    pdf_metadata['label'] = pdf_metadata['label_id'].map(self._class_map)

    # Add audio-specific metadata and save to csv
    pdf_metadata = self._add_audio_metadata_to_pdf_metadata(pdf_metadata)
    pdf_metadata.to_csv(self._metadata_file_path)
