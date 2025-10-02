"""
This is a configuration file used to forward desired parameters to
the different modules used during training.
"""
import os
import random
import socket
import string


cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

# Assign job_id to the run
if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))


DATA = os.environ['DATA_DIR']

dataset_dir = os.path.join(
    DATA,
    "ImpulseSeparationScenes" # or to whatever name you have given to the dataset folder
)

dataset_dir_reverb = os.path.join(
    DATA,
    "ImpulseSeparationScenes_reverb" # or to whatever name you have given to the dataset folder
)

csv_file = os.path.join(dataset_dir, "metadata.csv")

csv_file_reverb = os.path.join(dataset_dir_reverb, "metadata.csv")

sr=44100

stft_params = {
    "nfft": 2048,
    "overlap": 0.75,
    "window_size": 2048,
    "window": None,
    "center": True
}

common_parameters = {
    "exp_dir": os.path.join(
        os.environ['REPO_SAVE'],
        "is3",
        "baselines",
        "convtasnet",
        "results"
    ),
    "dataset": {
        "dataset_dir": dataset_dir_reverb,
        "csv_file": csv_file_reverb,
        "random_gain": True,
        "sampling_rate": sr,
    },
    "optim": {
        "lr": 0.001,
        "betas": (
            0.9,
            0.999),        
        "weight_decay": 0.0001,
        'batch_size': 4,
        "epochs": 150,
        "patience": 30,
        "lr_scheduler": None,  # "OneCycleLR",
        "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
                              "div_factor": 10, "pct_start": 0.15},   
        "early_stop": True,     
        },
    "process": {
        "num_workers": 4,
        "prefetch": 2,
        "devices": int(
            os.environ["SLURM_GPUS_ON_NODE"]) if "SLURM_GPUS_ON_NODE" in os.environ else 1,
        "num_nodes": int(
            os.environ["SLURM_NNODES"]) if slurm else 1,
        },
    "job_id": job_id,
    "seed": 10,
    "sr": sr,
    "checkpoint_path": None,
}
common_parameters["stft_params"] = stft_params

conf = {
    "001": {
        "dataset":{
            "sampling_rate": 44100,
        },
    },
    "002": {
        "dataset":{
            "sampling_rate": 8000,
        },
        "optim": {
            "batch_size": 8,
        },
    },
    "003": {
        "dataset":{
            "sampling_rate": 8000,
        },
        "optim": {
            "batch_size": 16,
        },
    },    
    "004": {
        "dataset":{
            "sampling_rate": 8000,
        },
        "optim": {
            "batch_size": 32,
        },
    },    
}
