from ast import literal_eval
import os
import sys
import pandas as pd

import config
from utils import SaltWrapper

sys.path.append(os.path.abspath('../pre_processing_datasets'))

import data_wrappers

DATA_DIR_PATH = os.path.abspath('../datasets/data_preprocessed')

COLS_TO_KEEP = ['file_path', 'n_channels', 'sampling_rate', 'duration']

class DatasetLoader():
  """Abstract dataset loader
  """
  __all__ = [
    'Esc50Wrapper',
    'Nonspeech7kWrapper',
    'RealisedWrapper',
    'VocalSoundWrapper',
    'FreesoundOneShotPercussiveWrapper',
    'Arte',
    'LitisRouenWrapper',
    'CochlSceneWrapper',
    'Cas2023Wrapper',
    'VehicleInteriorSoundWrapper',
    'Dcase2018Wrapper',
  ]

  @property
  def module(self):
    return self._module

  @property
  def cls(self):
    return self._cls

  @property
  def pdf_metadata(self):
    return self._pdf_metadata

  @property
  def label_col(self):
    return self._label_col

  @property
  def root_dir(self):
    return self._root_dir

  @pdf_metadata.setter
  def pdf_metadata(self, value):
    self._pdf_metadata = value

  def __init__(self, root_dir, cls, label_col, cols_to_keep, impulse_type = None, background_type=None):

    if root_dir is not None: # default db dir in adasp_data_management
      assert os.path.isdir(root_dir), f'{root_dir} is not a directory!'

    self._root_dir = root_dir
    self.salt_wrapper = SaltWrapper()
    self._module = data_wrappers
    self._label_col = label_col
    self._cols_to_keep = cols_to_keep + [label_col]
    self._std_label_col = 'std_label'
    self.impulse_type = impulse_type
    self.background_type = background_type
    self._cls = cls

    self.load_metadata_file()

  def load_metadata_file(self):
    # - Init pdf_metadata
    obj = getattr(self.module, self.cls)(root_dir=self._root_dir)

    self._pdf_metadata = obj.pdf_metadata.copy()
    
    # replace '' in duration with 0
    self._pdf_metadata['duration'] = self._pdf_metadata['duration'].replace('', 0).astype(float)

    # Filter out corrupted files
    self._pdf_metadata = self._pdf_metadata.loc[
      self._pdf_metadata['duration'] > 0].reset_index(drop=True)

    if self._root_dir is None:
      self._root_dir = obj.root_dir

    assert self._label_col in self.pdf_metadata.columns, \
      f'"{self.label_col}" not found in metadata columns'

    # - Remove unecessary columns
    for col in self._cols_to_keep:
      if col not in self._pdf_metadata.columns:
        raise ValueError(f'"{col}" not found in metadata columns')
    self._pdf_metadata = self._pdf_metadata[self._cols_to_keep]

    # - Rename label col
    if self._label_col != 'dataset_label':
      self._pdf_metadata.rename(columns={self._label_col: 'dataset_label'},
                                inplace=True)

    # Add impulse type ("natural"/"synthetic") if needed
    if self.impulse_type is not None:
      self._pdf_metadata['impulse_type'] = self.impulse_type
      
    if self.background_type is not None:
      self._pdf_metadata['background_type'] = self.background_type


  def map_to_salt(self, ontology):
    if ontology == 'event':
      mapper = self.salt_wrapper.event_mapper
    elif ontology == 'scene':
      mapper = self.salt_wrapper.scene_mapper
    else:
      raise ValueError('Unknown argument')

    dataset_labels = self._pdf_metadata['dataset_label'].unique().tolist()

    map_dict = {}
    for lbl in dataset_labels:
      map_dict[lbl] = mapper.get_std_label_from_dataset_label(lbl)

    self._pdf_metadata[self._std_label_col] = self._pdf_metadata[
      'dataset_label'].map(map_dict)

class BackgroundLoader():
  """Data loader for background sounds
  """
  def __init__(self):
    self.datasets = [
      cls() for cls in DatasetLoaderRegistry.get_background_classes()
    ]

    self.pdf_metadata = None
    self.generate_metadata()

  def generate_metadata(self):
    for cls in self.datasets:
      # - Add dataset name
      cls.pdf_metadata['dataset'] = cls.cls
      
      # Remove rows where duration is '' (no more audio files)
      cls.pdf_metadata = cls.pdf_metadata[cls.pdf_metadata['duration'] != '']      

      # - Add SALT labels
      cls.map_to_salt(ontology='scene')
      
      # replace '' in duration with 0
      cls.pdf_metadata['duration'] = cls.pdf_metadata['duration'].replace('', 0).astype(float)

      self.pdf_metadata = pd.concat([self.pdf_metadata, cls.pdf_metadata],
                                    ignore_index=True)


class ForegroundLoader():
  """Data loader for foreground (impulse) sounds
  """
  def __init__(self):
    self.datasets = [
      cls() for cls in DatasetLoaderRegistry.get_foreground_classes()
    ]

    self.pdf_metadata = None
    self.generate_metadata()

  def generate_metadata(self):
    for cls in self.datasets:
      # - Add dataset name
      cls.pdf_metadata['dataset'] = cls.cls
      
      # Remove rows where duration is '' (no more audio files)
      cls.pdf_metadata = cls.pdf_metadata[cls.pdf_metadata['duration'] != '']

      # - Keep only necessary labels
      # cls.pdf_metadata = cls.pdf_metadata.loc[
      #   cls.pdf_metadata['dataset_label'].isin(cls.salt_wrapper.impulse_labels)]
      
      dataset_labels = cls.salt_wrapper.event_mapper.map_df["dataset_label"].unique().tolist()
      
      cls.pdf_metadata = cls.pdf_metadata.loc[
        cls.pdf_metadata['dataset_label'].isin(dataset_labels)]      

      # - Add SALT labels
      cls.map_to_salt(ontology='event')

      self.pdf_metadata = pd.concat([self.pdf_metadata, cls.pdf_metadata],
                                    ignore_index=True)


class Esc50Wrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'esc50_only_impulses', 'ESC-50-master'),  # Change if needed
      cls='ESC50',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      impulse_type='natural'
    )

class Nonspeech7kWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'nonspeech7k_only_impulses'),  # Change if needed
      cls='Nonspeech7k',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      impulse_type='natural'
    )

class RealisedWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'realised_only_impulses'),  # Change if needed
      cls='ReaLISED',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      impulse_type='natural'
    )

class VocalSoundWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'vocalsound_only_impulses'),  # Change if needed
      cls='VocalSound',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      impulse_type='natural'
    )

# class FreesoundOneShotPercussiveWrapper(DatasetLoader):
#   def __init__(self):
#     super().__init__(
#       root_dir=os.path.join(DATA_DIR_PATH, 'freesound_oneshot_percussive_only_impulses'),  # Change if needed
#       cls='FreesoundOneShotPercussive',
#       label_col='tags',
#       cols_to_keep=COLS_TO_KEEP,
#       impulse_type='natural'
#     )

    # # Expand "dataset_label" col
    # self.pdf_metadata['dataset_label'] = self.pdf_metadata[
    #   'dataset_label'].apply(literal_eval)
    # self.pdf_metadata = self.pdf_metadata.explode('dataset_label')

class ArteWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'arte_impulse_free'),  # Change if needed
      cls='Arte',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      background_type='natural'
    )

class CochlSceneWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'cochlscene_impulse_free'),  # Change if needed
      cls='CochlScene',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      background_type='natural'
    )

class Cas2023Wrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'cas2023_impulse_free'),  # Change if needed
      cls='Cas2023',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      background_type='natural'
    )

class VehicleInteriorSoundWrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'visc_son_impulse_free'),  # Change if needed
      cls='VehicleInteriorSound',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      background_type='natural'
    )

class Dcase2018Wrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(DATA_DIR_PATH, 'dcase2018_impulse_free', 'TUT-urban-acoustic-scenes-2018-development'),  # Change if needed
      cls='Dcase2018',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      background_type='natural'
    )

# Loader Registry
class DatasetLoaderRegistry:
  """Registry to associate dataset wrappers with loaders"""
  _background_classes = []
  _foreground_classes = []

  @classmethod
  def register_background(cls, wrapper_class):
    cls._background_classes.append(wrapper_class)

  @classmethod
  def register_foreground(cls, wrapper_class):
    cls._foreground_classes.append(wrapper_class)

  @classmethod
  def get_background_classes(cls):
    return cls._background_classes

  @classmethod
  def get_foreground_classes(cls):
    return cls._foreground_classes


# Register Wrappers
DatasetLoaderRegistry.register_foreground(Esc50Wrapper)
DatasetLoaderRegistry.register_foreground(Nonspeech7kWrapper)
DatasetLoaderRegistry.register_foreground(RealisedWrapper)
DatasetLoaderRegistry.register_foreground(VocalSoundWrapper)

DatasetLoaderRegistry.register_background(ArteWrapper)
DatasetLoaderRegistry.register_background(CochlSceneWrapper)
DatasetLoaderRegistry.register_background(Cas2023Wrapper)
DatasetLoaderRegistry.register_background(VehicleInteriorSoundWrapper)
DatasetLoaderRegistry.register_background(Dcase2018Wrapper)
