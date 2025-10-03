import os 

ESC50 = "path_to_ESC-50"  # Please download and set the path to ESC-50 dataset

NONSPEECH7K = "path_to_NonSpeech7K"

DRUMSAMPLES = "path_to_drum_samples"

FREESOUND = "path_to_freesound_dataset"

datasets_to_clean = {
    "ESC-50": {
        "src_dir": ESC50,
        "dst_dir": "path_to_save_cleaned_ESC-50",
    },
    "NonSpeech7K": {
        "src_dir": NONSPEECH7K,
        "dst_dir": "path_to_save_cleaned_NonSpeech7K",
    },
    "DrumSamples": {
        "src_dir": DRUMSAMPLES,
        "dst_dir": "path_to_save_cleaned_DrumSamples",
    },
    "Freesound": {
        "src_dir": FREESOUND,
        "dst_dir": "path_to_save_cleaned_Freesound",
    },
}