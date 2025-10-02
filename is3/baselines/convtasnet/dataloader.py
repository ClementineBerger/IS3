import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchaudio

class ImpulseSeparationScenesDataset(Dataset):
  """
    Initializes the data loader for processing audio datasets.
    Parameters
    ----------
    root_dir : str
      The root directory containing the audio files.
    csv_file : str
      Path to the CSV file containing metadata about the audio files.
    subset : str, optional
      The subset of data to use, either "train", "validation", or "test". 
      Default is "test".
    random_gain : bool, optional
      If True, applies random gain to the audio data during testing. 
      Default is False.
    Attributes
    ----------
    root_dir : str
      The root directory containing the audio files.
    csv_file : str
      Path to the CSV file containing metadata about the audio files.
    pdf_metadata : pandas.DataFrame
      DataFrame containing metadata filtered by the specified subset.
    subset : str
      The subset of data being used.
    random_gain : bool
      Indicates whether random gain is applied during testing.
    overall_maximum : float
      The maximum gain ratio computed from the metadata.
    bkg_path_list : list of str
      List of file paths for background audio files.
    impulse_path_list : list of str
      List of file paths for impulse audio files.
    mix_path_list : list of str
      List of file paths for mixture audio files.
    gains : list of float
      List of gain ratios computed from the metadata.
    pre_compute_random_gains : numpy.ndarray
      Precomputed random gains for testing, if `random_gain` is True 
      and `subset` is "test".
  """  
  
  def __init__(
    self,
    root_dir, 
    csv_file,
    subset="train",
    random_gain=False,
    sampling_rate=44100,
  ):
    
    self.root_dir = root_dir
    self.csv_file = csv_file
    self.pdf_metadata = pd.read_csv(csv_file)
    self.pdf_metadata = self.pdf_metadata[self.pdf_metadata['subset'] == subset]
    self.subset = subset
    self.random_gain=random_gain
    self.sampling_rate = sampling_rate
    
    gains = self.pdf_metadata['dba_normalization_gain'] / self.pdf_metadata['normalization_gain']
    
    self.overall_maximum = np.max(gains)
    
    # set path list
    self.bkg_path_list = [
      os.path.join(self.root_dir, bkg_path)
      for bkg_path in self.pdf_metadata['background_audio_path'].values]
    self.impulse_path_list = [
      os.path.join(self.root_dir, impulse_path)
      for impulse_path in self.pdf_metadata['impulse_audio_path'].values]
    self.mix_path_list = [
      os.path.join(self.root_dir, mix_path)
      for mix_path in self.pdf_metadata['mixture_audio_path'].values]
    
    # gains list
    self.gains = gains.values.tolist()
    
    # if testing, seeding so that every gains are the same
    if subset=="test" and self.random_gain :
      np.random.seed(28)
      self.pre_compute_random_gains = np.random.uniform(low=1e-5, high=0.999, size=len(self.gains))  #test sur range de gains, à changer ensuite
    
  def __len__(self):
    return len(self.pdf_metadata)
  
  def __getitem__(self, idx):
    
    bkg_path = self.bkg_path_list[idx]
    impulse_path = self.impulse_path_list[idx]
    mix_path = self.mix_path_list[idx]
    gain = self.gains[idx]
    
    bkg_waveform, sr_init = torchaudio.load(bkg_path)
    impulse_waveform, _ = torchaudio.load(impulse_path)
    mix_waveform, _ = torchaudio.load(mix_path)
    
    if sr_init != self.sampling_rate:
      bkg_waveform = torchaudio.functional.resample(bkg_waveform, sr_init, self.sampling_rate)
      impulse_waveform = torchaudio.functional.resample(impulse_waveform, sr_init, self.sampling_rate)
      mix_waveform = torchaudio.functional.resample(mix_waveform, sr_init, self.sampling_rate)
    
    if not self.random_gain:
      bkg_waveform = bkg_waveform*gain/self.overall_maximum
      impulse_waveform = impulse_waveform*gain/self.overall_maximum
      mix_waveform = mix_waveform*gain/self.overall_maximum
    else: 
      if self.subset == "test":
        rd_gain = self.pre_compute_random_gains[idx]
      else:
        rd_gain = np.random.uniform(low=1e-5, high=0.999)
      bkg_waveform = bkg_waveform*rd_gain
      impulse_waveform = impulse_waveform*rd_gain 
      mix_waveform = mix_waveform*rd_gain
    
    return mix_waveform.squeeze(0), impulse_waveform.squeeze(0), bkg_waveform.squeeze(0)