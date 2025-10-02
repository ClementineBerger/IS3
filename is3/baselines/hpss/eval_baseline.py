

from hpss import HarmonicPercussiveDecomposition
from dataloader import ImpulseSeparationScenesDataset
from config import conf

import torch
from torch.utils.data import DataLoader

from is3.metrics import compute_si_sdr

import numpy as np

import pandas as pd

import json

import os
import argparse
from pprint import pprint
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--conf_id", default="001", type=str,  # change default for debbug
                    help="Conf tag, used to get the right config")
parser.add_argument("--dataset", default="random", type=str,  #original or random
                    help="Using the original (normalized) test set or the random gain dataset")

RENDERING_DATA = os.path.join(
    os.environ['AUDIBLE_DATA'],
    "rendering",
    "datasets"
)

# RENDERING_DATA = os.path.join(
#   os.environ['MY_DATA']
# )

dataset_dir = os.path.join(
    RENDERING_DATA,
    "impulsive_sound_rejection",
    "ImpulseSeparationScenes"  #_reverb
)

csv_file = os.path.join(dataset_dir, "metadata.csv")

# Analysis params
nfft = 2048
sr = 44100
overlap = 0.75

def main(conf_id, dataset):
    
  if dataset=="original":
    random_gain = False 
  elif dataset=="random":
    random_gain = True
  
  test_dataset = ImpulseSeparationScenesDataset(
    root_dir=dataset_dir,
    csv_file=csv_file,
    random_gain=random_gain,
    subset="test",
  )
  
  test_dataloader = DataLoader(test_dataset,
                               batch_size=1,  # hpss -> 1 audio par 1 audio
                               shuffle=False,
                               num_workers=2,
                               drop_last=False,
                               pin_memory=True)  
  
  
  margin = conf[conf_id]["margin"]
  
  hpss = HarmonicPercussiveDecomposition(
    nfft=nfft,
    window_size=nfft,
    overlap=overlap,
    margin=margin
  )
  
  all_si_sdr_impulse = []
  all_sdr_impulse = []
  all_si_sdr_impulse_no_silence = []
  all_sdr_impulse_no_silence = []
  
  all_si_sdr_bkg = []
  all_sdr_bkg = []
  all_si_sdr_bkg_no_silence = []
  all_sdr_bkg_no_silence = []
  
  for data in tqdm(test_dataloader):
    mix_target, imp_target, bkg_target = data 

    imp_pred, bkg_pred, S_p, S_h = hpss.forward(mix_target.numpy())
    
    imp_pred = torch.tensor(imp_pred)
    bkg_pred = torch.tensor(bkg_pred)
    
    all_si_sdr_impulse.append(compute_si_sdr(imp_pred, imp_target, scaling=True, remove_silences=False))
    all_sdr_impulse.append(compute_si_sdr(imp_pred, imp_target, scaling=False, remove_silences=False))
    all_si_sdr_impulse_no_silence.append(compute_si_sdr(imp_pred, imp_target, scaling=True, remove_silences=True, frame_size=1024, delta=1e-2))
    all_sdr_impulse_no_silence.append(compute_si_sdr(imp_pred, imp_target, scaling=False, remove_silences=True, frame_size=1024, delta=1e-2))      
    
    all_si_sdr_bkg.append(compute_si_sdr(bkg_pred, bkg_target, scaling=True, remove_silences=False))
    all_sdr_bkg.append(compute_si_sdr(bkg_pred, bkg_target, scaling=False, remove_silences=False))    
    all_si_sdr_bkg_no_silence.append(compute_si_sdr(bkg_pred, bkg_target, scaling=True, remove_silences=True, frame_size=1024, delta=1e-2))
    all_sdr_bkg_no_silence.append(compute_si_sdr(bkg_pred, bkg_target, scaling=False, remove_silences=True, frame_size=1024, delta=1e-2))    
    
  results = {
    "imp":{
      "si_sdr": torch.concatenate(all_si_sdr_impulse).cpu().tolist(),
      "sdr": torch.concatenate(all_sdr_impulse).cpu().tolist(),
      "si_sdr_no_silence": torch.concatenate(all_si_sdr_impulse_no_silence).cpu().tolist(),
      "sdr_no_silence": torch.concatenate(all_sdr_impulse_no_silence).cpu().tolist()
    },
    "bkg":{
      "si_sdr": torch.concatenate(all_si_sdr_bkg).cpu().tolist(),
      "sdr": torch.concatenate(all_sdr_bkg).cpu().tolist(),
      "si_sdr_no_silence": torch.concatenate(all_si_sdr_bkg_no_silence).cpu().tolist(),
      "sdr_no_silence": torch.concatenate(all_sdr_bkg_no_silence).cpu().tolist()      
    },
  }
  
  saving_dir = os.path.join(
    os.environ["RENDERING_SAVE"],
    "studies",
    "006_impulsive_sound_rejection",
    "007_hpss_baseline",
    "results",
    conf_id,
    "evaluation"    
  )
  
  os.makedirs(saving_dir, exist_ok=True)  
  
  if dataset=="original":
    filename="metrics.json"
  elif dataset=="random":
    filename="metrics_random_gains.json"
  
  saving_path = os.path.join(
      saving_dir,
      filename
  )
  
  # Sauvegarder en format .json
  with open(saving_path, 'w') as f:
    json.dump(results, f)

  print("Done")

  return results  

if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  main(args["conf_id"], dataset=args["dataset"])