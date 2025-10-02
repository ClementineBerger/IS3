import torch
import torch.nn as nn
import torchaudio

import librosa

import os 

import yaml 

from model import ImpulseSoundRejection

root_dir = os.path.join(
  os.environ['AUDIBLE_DATA'],
  "rendering"
)

saved_models = os.path.join(
  root_dir, 
  "studies",
  "006_impulsive_sound_rejection",
  "005_df_model",
  "results",
)

class ModelWrapper(nn.Module):
  def __init__(self, conf_name, job_id=None):
    super().__init__()
    
    self.conf_name = conf_name
    self.job_id = job_id
    
    self.conf_path = os.path.join(
      saved_models,
      conf_name,
    )
    
    if self.job_id is None:
      list_dir = os.listdir(self.conf_path)
      try:
        list_dir.remove('evaluation')
      except BaseException:
        pass
      try:
        list_dir.remove('archive_models')
      except BaseException:
        pass
      job_id = list_dir[-1]  # on prend le dernier (le seul normalement)  
      
    if conf_name=="011":  
      job_id = "630354"
    elif conf_name=="012":
      job_id = "630358"
    elif conf_name=="012b":
      job_id = "630359"
    elif conf_name=="013":
      job_id = "630360"
    elif conf_name=="014":
      job_id = "630361"
    elif conf_name=="015":
      job_id = "647311"
    model_path = os.path.join(
        self.conf_path,
        job_id,
        "best_model.pth"
    )

    config_path = os.path.join(
        self.conf_path,
        job_id,
        "conf.yml"
    )

    with open(config_path, 'r') as f:
      self.conf = yaml.safe_load(f)   
      
    if 'common_encoder' not in self.conf['model']:
      self.conf['model']['common_encoder'] = True
      
    self.model = ImpulseSoundRejection(**self.conf['model'])
    
    self.model.load_state_dict(torch.load(model_path))
    
  def forward(self, x):
    # x : mixture waveform [B, T]
    
    
    y_i_spec, y_s_spec = self.model.forward_audio(x)
    
    y_i_spec = torch.view_as_complex(y_i_spec).transpose(-1, -2)
    y_s_spec = torch.view_as_complex(y_s_spec).transpose(-1, -2)    
    
    y_i = self.model.stft.inverse(y_i_spec, length=x.shape[-1])
    y_s = self.model.stft.inverse(y_s_spec, length=x.shape[-1])
    
    return y_i, y_s