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

erb_params = {
    "sr": sr, 
    "fft_size": stft_params["nfft"], 
    "n_bands": 24,
    "min_n_freqs": 2, 
    "normalized": False, 
    "min_mean_norm": -60,
    "max_mean_norm": -90, 
    "alpha": 0.99
}

feat_spec_params = {
    "n_feat": 256,   # 5376 Hz
    "alpha": 0.99, 
    "normalized": False,
    "min_unit_norm": 0.001,
    "max_unit_norm": 0.0001
}

common_parameters = {
    "exp_dir": os.path.join(
        os.environ['REPO_SAVE'],
        "is3",
        "is3",
        "results"
    ),
    "dataset": {
        "dataset_dir": dataset_dir,
        "csv_file": csv_file,
        "random_gain": False,
    },
    "model": {
        "conv_ch": 64,
        "emb_hidden_dim": 256,
        "lin_groups": 16,
        "enc_lin_groups": 32,
        "df_hidden_dim": 256,
        "df_num_layers": 2,
        "df_order": 5,
        "df_pathway_kernel_size_f": 5,
        "df_lookahead": 0,
        "conv_lookahead": 0,
        "conv_kernel_inp": (1, 3),
        "conv_kernel": (1, 3),
        "enc_num_layers": 1,
        "erb_num_layers": 2,
        "mask_pf": False,
        "df_n_iter": 1,
        "rnn_type": "gru",
        "gru_groups": 1,
        "trans_conv_type": "conv_transpose",
        "stft_params": stft_params,
        "erb_params": erb_params,
        "nb_erb": erb_params["n_bands"],
        "nb_df": feat_spec_params["n_feat"],
        "feat_spec_params": feat_spec_params,
        "erb_norm": True,
        "feat_spec_norm": True,
        "branch": ["erb", "df"],
        "common_encoder": True,
    },
    "loss": {
        "gamma_spec": 0.3,
        "f_mag_spec": 1000,
        "f_complex_spec": 1000,
        "f_under_spec": 1,
        "n_ffts": (256, 512, 1024),
        "gamma_mr": 0.3,
        "f_mag_mr": 500,
        "f_complex_mr": 500,
        "stft_params": stft_params,
        "weight_impulse": 1,
        "weight_stationary": 1,
        "weight_mixture": 1,
        "weight_stationary_when_impulse": None,
        "normalize": False,
    },
    "optim": {
        "lr": 0.001,
        "betas": (
            0.9,
            0.999),        
        "weight_decay": 0.0001,
        'batch_size': 32,
        "epochs": 50,
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
common_parameters["erb_params"] = erb_params
common_parameters["feat_spec_params"] = feat_spec_params

conf = {
    "014":{               # config "de base" 
        "dataset": {
            "dataset_dir": dataset_dir_reverb,
            "csv_file": csv_file_reverb,
            "random_gain": True,
        },       
        "optim":
            {
             "epochs": 150,   
        },      
        "training": {
        "lr_scheduler": None,
        },
        "loss":{
            "weight_impulse": 1,
            "weight_stationary": 5,
            "weight_mixture": 1,
        },     
        "stft_params": {
            "nfft": 2048,
            "window_size": 2048, 
        },
        "feat_spec_params": {
            "n_feat": 256,   # 5376 Hz
        },
        "erb_params": {
            "n_bands": 24,
        },
    },    
    "014_only_erb":{               # config "de base"
        "dataset": {
            "dataset_dir": dataset_dir_reverb,
            "csv_file": csv_file_reverb,
            "random_gain": True,
        },       
        "optim":
            {
             "epochs": 150,   
        },      
        "training": {
        "lr_scheduler": None,
        },
        "loss":{
            "weight_impulse": 1,
            "weight_stationary": 5,
            "weight_mixture": 1,
        },     
        "stft_params": {
            "nfft": 2048,
            "window_size": 2048, 
        },
        "feat_spec_params": {
            "n_feat": 256,   # 5376 Hz
        },
        "erb_params": {
            "n_bands": 24,
        },
        "model":{
            "branch": ["erb"],
            "common_encoder": False,
        }
    },   
    "014_only_erb_ce":{               # config "de base"
        "dataset": {
            "dataset_dir": dataset_dir_reverb,
            "csv_file": csv_file_reverb,
            "random_gain": True,
        },       
        "optim":
            {
             "epochs": 150,   
        },      
        "training": {
        "lr_scheduler": None,
        },
        "loss":{
            "weight_impulse": 1,
            "weight_stationary": 5,
            "weight_mixture": 1,
        },     
        "stft_params": {
            "nfft": 2048,
            "window_size": 2048, 
        },
        "feat_spec_params": {
            "n_feat": 256,   # 5376 Hz
        },
        "erb_params": {
            "n_bands": 24,
        },
        "model":{
            "branch": ["erb"],
            "common_encoder": True,
        }
    },    
    "014_only_df":{               # config "de base"
        "dataset": {
            "dataset_dir": dataset_dir_reverb,
            "csv_file": csv_file_reverb,
            "random_gain": True,
        },       
        "optim":
            {
             "epochs": 150,   
        },      
        "training": {
        "lr_scheduler": None,
        },
        "loss":{
            "weight_impulse": 1,
            "weight_stationary": 5,
            "weight_mixture": 1,
        },     
        "stft_params": {
            "nfft": 2048,
            "window_size": 2048, 
        },
        "feat_spec_params": {
            "n_feat": 256,   # 5376 Hz
        },
        "erb_params": {
            "n_bands": 24,
        },
        "model":{
            "branch": ["df"],
            "common_encoder": False,
        },
    },    
    "014_only_df_ce":{               # config "de base"
        "dataset": {
            "dataset_dir": dataset_dir_reverb,
            "csv_file": csv_file_reverb,
            "random_gain": True,
        },       
        "optim":
            {
             "epochs": 150,   
        },      
        "training": {
        "lr_scheduler": None,
        },
        "loss":{
            "weight_impulse": 1,
            "weight_stationary": 5,
            "weight_mixture": 1,
        },     
        "stft_params": {
            "nfft": 2048,
            "window_size": 2048, 
        },
        "feat_spec_params": {
            "n_feat": 256,   # 5376 Hz
        },
        "erb_params": {
            "n_bands": 24,
        },
        "model":{
            "branch": ["df"],
            "common_encoder": True,
        },
    },                         
}
