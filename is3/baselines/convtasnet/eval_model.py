"""
Evaluation of the model.
"""

from dataloader import ImpulseSeparationScenesDataset
from model_wrapper import ModelWrapper

import torch
from torch.utils.data import DataLoader

from is3.metrics import compute_si_sdr, compute_metrics_batch

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
parser.add_argument("--dataset", default="original", type=str,  #original or random
                    help="Using the original (normalized) test set or the random gain dataset")


RENDERING_DATA = os.path.join(
    os.environ['AUDIBLE_DATA'],
    "rendering",
    "datasets"
)

# RENDERING_DATA = os.path.join(
#   os.environ["MY_DATA"],
# )

dataset_dir = os.path.join(
    RENDERING_DATA,
    "impulsive_sound_rejection",
    "ImpulseSeparationScenes_reverb"
)

csv_file = os.path.join(dataset_dir, "metadata.csv")


def main(conf_id, dataset):
  
  model = ModelWrapper(conf_name=conf_id)
  model.eval()
  if torch.cuda.is_available():
    model = model.cuda()
    
  if dataset=="original":
    random_gain = False 
  elif dataset=="random":
    random_gain = True
  
  test_dataset = ImpulseSeparationScenesDataset(
    root_dir=dataset_dir,
    csv_file=csv_file,
    random_gain=random_gain,
    subset="test",
    sampling_rate=model.conf["dataset"]["sampling_rate"],
  )
  
  test_dataloader = DataLoader(test_dataset,
                               batch_size=4,
                               shuffle=False,
                               num_workers=4,
                               drop_last=False,
                               pin_memory=True)  
  
  # # save indexes in metadata in order to retrieve the audios info if necessary
  # indexes_in_metadata = []
  
  all_si_sdr_impulse = []
  all_sdr_impulse = []
  all_si_sdr_impulse_no_silence = []
  all_sdr_impulse_no_silence = []
  
  all_si_sdr_bkg = []
  all_sdr_bkg = []
  # all_si_sdr_bkg_no_silence = []
  # all_sdr_bkg_no_silence = []
  
  all_si_sdr_mix = []  # à ne pas calculer pour HPSS car la reconstruction est forcément parfaite dans ce cas
  all_sdr_mix = []  
  # all_si_sdr_mix_no_silence = []
  # all_sdr_mix_no_silence = []
  
  
  all_si_sir_imp = []
  all_si_sir_bkg = []
  
  all_si_sar_imp = []
  all_si_sar_bkg = []
  
  for data in tqdm(test_dataloader):
    mix_target, imp_target, bkg_target = data 
    
    with torch.no_grad():
      if torch.cuda.is_available():
        mix_target = mix_target.cuda()
        imp_target = imp_target.cuda()
        bkg_target = bkg_target.cuda()
    
      imp_pred, bkg_pred = model.forward(
        x = mix_target
      )
      
      mix_pred = imp_pred + bkg_pred     
      
      # SI-SDR and SDR
      all_si_sdr_impulse.append(compute_si_sdr(imp_pred, imp_target, scaling=True, remove_silences=False))
      all_sdr_impulse.append(compute_si_sdr(imp_pred, imp_target, scaling=False, remove_silences=False))
      all_si_sdr_impulse_no_silence.append(compute_si_sdr(imp_pred, imp_target, scaling=True, remove_silences=True, frame_size=1024, delta=1e-2))
      all_sdr_impulse_no_silence.append(compute_si_sdr(imp_pred, imp_target, scaling=False, remove_silences=True, frame_size=1024, delta=1e-2))      
      
      all_si_sdr_bkg.append(compute_si_sdr(bkg_pred, bkg_target, scaling=True, remove_silences=False))
      all_sdr_bkg.append(compute_si_sdr(bkg_pred, bkg_target, scaling=False, remove_silences=False))    
      # all_si_sdr_bkg_no_silence.append(compute_si_sdr(bkg_pred, bkg_target, scaling=True, remove_silences=True, frame_size=1024, delta=1e-2))
      # all_sdr_bkg_no_silence.append(compute_si_sdr(bkg_pred, bkg_target, scaling=False, remove_silences=True, frame_size=1024, delta=1e-2))      
      
      all_si_sdr_mix.append(compute_si_sdr(mix_pred, mix_target, scaling=True, remove_silences=False))
      all_sdr_mix.append(compute_si_sdr(mix_pred, mix_target, scaling=False, remove_silences=False))  
      # all_si_sdr_mix_no_silence.append(compute_si_sdr(mix_pred, mix_target, scaling=True, remove_silences=True, frame_size=1024, delta=1e-2))
      # all_sdr_mix_no_silence.append(compute_si_sdr(mix_pred, mix_target, scaling=False, remove_silences=True, frame_size=1024, delta=1e-2))       
      
      # SI-SIR and SI-SAR
      reference_signals = torch.concatenate((imp_target.unsqueeze(-1), bkg_target.unsqueeze(-1)), dim=-1)
      
      ## Impulse
      _, si_sir, si_sar = compute_metrics_batch(
        estimated_signal=imp_pred,
        reference_signals=reference_signals,
        source_index=0,
        scaling=True
      )
      
      all_si_sir_imp.append(si_sir)
      all_si_sar_imp.append(si_sar)
      
      ## Bkg
      _, si_sir, si_sar = compute_metrics_batch(
        estimated_signal=bkg_pred,
        reference_signals=reference_signals,
        source_index=1,
        scaling=True
      )      
      
      all_si_sir_bkg.append(si_sir)
      all_si_sar_bkg.append(si_sar)      
    
  results = {
    "imp":{
      "si_sdr": torch.concatenate(all_si_sdr_impulse).cpu().tolist(),
      "sdr": torch.concatenate(all_sdr_impulse).cpu().tolist(),
      "si_sdr_no_silence": torch.concatenate(all_si_sdr_impulse_no_silence).cpu().tolist(),
      "sdr_no_silence": torch.concatenate(all_sdr_impulse_no_silence).cpu().tolist(),
      "si_sir": torch.concatenate(all_si_sir_imp).cpu().tolist(), 
      "si_sar":torch.concatenate(all_si_sar_imp).cpu().tolist()
    },
    "bkg":{
      "si_sdr": torch.concatenate(all_si_sdr_bkg).cpu().tolist(),
      "sdr": torch.concatenate(all_sdr_bkg).cpu().tolist(),
      # "si_sdr_no_silence": torch.concatenate(all_si_sdr_bkg_no_silence).cpu().tolist(),
      # "sdr_no_silence": torch.concatenate(all_sdr_bkg_no_silence).cpu().tolist(),    
      "si_sir": torch.concatenate(all_si_sir_bkg).cpu().tolist(), 
      "si_sar":torch.concatenate(all_si_sar_bkg).cpu().tolist()        
    },
    "mix":{
      "si_sdr": torch.concatenate(all_si_sdr_mix).cpu().tolist(),
      "sdr": torch.concatenate(all_sdr_mix).cpu().tolist(),
      # "si_sdr_no_silence": torch.concatenate(all_si_sdr_mix_no_silence).cpu().tolist(),
      # "sdr_no_silence": torch.concatenate(all_sdr_mix_no_silence).cpu().tolist()      
    }
  }
  
  saving_dir = os.path.join(
    os.environ["RENDERING_SAVE"],
    "studies",
    "006_impulsive_sound_rejection",
    "012_convtasnet",
    "results",
    model.conf["conf_id"],
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

  main(str(args["conf_id"]), str(args['dataset']))