import torch
import torch.nn as nn
import torchaudio

import librosa

import os 

import yaml 

from asteroid.models import ConvTasNet


saved_models = os.path.join(
  os.environ['REPO_SAVE'],
  "is3",
  "baselines",
  "convtasnet",
  "results"
)

# jz_path_to_models = os.path.join(
#   os.environ["HOME"],
#   "mnt",
#   "scratch_jeanzay",
#   "audible",
#   "protected",
#   "dev",
#   "rendering",
#   "studies",
#   "006_impulsive_sound_rejection",
#   "012_convtasnet",
#   "results", 
# ) 

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
      job_id = list_dir[-1] 
      
    if conf_name=="001" :
      job_id = "1147875" #"415825" 
    elif conf_name=="002": 
      job_id = "415882"
      
    model_path = os.path.join(
        self.conf_path,
        job_id,
        # "best_model.pth"
        "checkpoint",
        "last.ckpt"
    )

    config_path = os.path.join(
        self.conf_path,
        job_id,
        "conf.yml"
    )

    with open(config_path, 'r') as f:
      self.conf = yaml.safe_load(f)   
      
    # Setting model
    if self.conf["dataset"]["sampling_rate"] == 44100:
      self.model = ConvTasNet(
        n_src=2,
        sample_rate=44100,
        causal=True,
        n_blocks=10,
        kernel_size=20,
        stride=10,
        # n_repeats=4,
        # nb_chan=256,
        # skip_chan=256,
      )
    else:
      self.model = ConvTasNet(
        n_src=2,
        causal=True,
        sampling_rate=self.conf["dataset"]["sampling_rate"],
        norm_type="cLN",
      )
    
    ckpt = torch.load(model_path)
    state_dict = ckpt["state_dict"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    self.model.load_state_dict(new_state_dict)
    
  def forward(self, x):
    
    prediction = self.model.forward(x)
    y_i_predicted = prediction[..., 0, :]
    y_s_predicted = prediction[..., 1, :]
    
    return y_i_predicted, y_s_predicted