# IS³ : Generic Impulsive-Stationary Sound Separation in Acoustic Scenes using Deep Filtering 

> **Authors**: C. Berger, P. Stamatiadis, R. Badeau, S. Essid 

Welcome to the IS³ repository! This repository contains the code and resources for the paper:
"IS³ : Generic Impulsive-Stationary Sound Separation in Acoustic Scenes using Deep Filtering", presented at WASPAA 2025.

## Abstract

We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics.

To this end, we introduce IS³, a neural network designed for Impulsive–Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic–Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics.

## Links

[:loud_sound: Audio examples](https://clementineberger.github.io/IS3/audio)

[:page_facing_up:]() [Paper](https://telecom-paris.hal.science/hal-05228563v2/) 


## Set-up


### Environment variables

You need to set the following environment variables:

- `DATA_DIR`: path to the root directory where the datasets will be generated and stored

- `REPO_SAVE`: path to your repository folder (where you cloned this repository) or a mirrored one (used to save the results of the experiments)

## IS³


## Data generation

We provide a data generation pipeline that creates training, validation and test sets for the impulsive-stationary sound separation task. The pipeline uses existing datasets of isolated impulsive sounds and stationary sounds, combines them with various augmentations and generates mixtures with their corresponding ground truth sources.

Here are the datasets we used in the paper (you can use your own datasets as well):
- Background sounds: Dcase2018 Task 1, CochlScene, Arte, Cas2023, LitisRouen
- Isolated events: ESC-50, ReaLISED, VocalSound, FreesoundOneShotPercussive, Nonspeech7k

For both types of sounds, we also generated synthetic sounds to increase the diversity of the training data.  

### Synthetic impulsive sounds

The code to generate synthetic impulsive sounds can be found in `data/synthetic_impulse_sounds/`. You can modify the parameters in `data/synthetic_impulse_sounds/config.py` to change the characteristics of the generated sounds.

We generate 3 types of synthetic impulsive sounds:
- chirp,
- harmonic summation,
- AR filtering of white noise modulated by asymmetric Gaussian envelopes.

The script `data/synthetic_impulse_sounds/generate_dataset.py` generates the synthetic impulsive sounds dataset and saves it in the specified output directory.

### Synthetic stationary backgrounds

We generate synthetic stationary backgrounds by applying various types of augmentations to pink noise to reproduce ventilation/wind-like sounds. The code can be found in `data/synthetic_stationary_sounds/`. You can modify the parameters in `data/synthetic_stationary_sounds/config.py` to change the characteristics of the generated sounds.

We use the `audiomentation` library to apply the augmentations (EQ, gain variations etc.) and in particular reveberberation using the `ApplyImpulseResponse` transform. This transform requires a list of path to impulse response files. In the paper, we used a mix of several datasets (OpenAIR, MARDY, EchoThief, etc.). You will need to provide your own list of impulse response files.

The script `data/synthetic_stationary_backgrounds/generate_dataset.py` generates the synthetic stationary backgrounds dataset and saves it in the specified output directory.

### Pre-processing existing datasets

Existing datasets chosen for the impulsive-stationary sound separation task may require some pre-processing to ensure that the sounds are suitable for the task. The background sounds should be impulsive event-free and the isolated impulsive sounds should be genuinely impulsive.

Before running the pre-processing scripts, you need to set the paths to the datasets in the configuration files.

#### Background sounds
The code to pre-process the existing datasets can be found in `data/preprocess_datasets/stationary_backgrounds`.

The idea of this pre-processings is to remove segments containing impulsive events from the background sounds datasets and reconnect the remaining segments.
To detect those impulsive events, we combine onset detection and Gabor decomposition using the Multi-Gabor dictionaries and procedures described in \[2\] and implemented in the Matlab/Octave toolbox `LTFAT` (https://ltfat.github.io/) with the python library `Oct2Py`. You will therefore need an Octave installation to run the scripts (see `data/preprocess_datasets/README.md`).

- `impulse_dection.py`: define a class to detect impulsive events in an audio signal using onset detection and Gabor decomposition.
- `utils.py`: utility functions for loading and saving audio files in a new mirrored folder.
- `clean.py`: script to remove impulsive events from a given dataset and save the cleaned audio files in a new mirrored folder.

#### Isolated impulsive sounds
The code to pre-process the existing datasets can be found in `data/preprocess_datasets/impulse_sounds`.

- `checking_impulse_sound.py`: define a class to check if a sound is impulsive based on its energy and silence proportion.
- `config.py`: configuration file with the paths to the used datasets.
- `removing_script.py`: script to remove non-impulsive sounds from a given dataset.
- `utils.py`: utility functions for loading and saving audio files in a new mirrored folder.

## Baselines

### Harmonic-Percussive Source Separation (HPSS) 

Using Librosa's implementation of HPSS using two different settings of the margin parameter $p_m$,

$$ M_p(t,f) = \frac{X_p(t, k)}{X_h(t,k) + \epsilon} \geq p_m $$

and we use the percussive mask to extract the impulsive source and its complement to extract the stationary source.

You can find the configuration file in `baselines/HPSS/config.py`. The configurations used in the paper are "001" with `margin=1` and "003" with `margin=2`.

### Nongpiur's wavelet filtering \[1\]
Implementation of the wavelet filtering method proposed by R. C. Nongpiur in \[1\].

You can find the configuration file in `baselines/wavelet/config.py`. The configuration used in the paper is "002". 

### Conv-TasNet

Based on `asteroid`'s implementation of Conv-TasNet \[2\], we adapted the model to perform source separation at a sampling rate of 44.1 kHz.

## Evaluation


## Cite us

If you find this work useful in your research, please consider citing:

```
@inproceedings{berger:hal-05228563,
  TITLE = {{IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering}},
  AUTHOR = {Berger, Cl{\'e}mentine and Stamatiadis, Paraskevas and Badeau, Roland and Essid, Slim},
  BOOKTITLE = {{IEEE Workshop on Applications of Signal Processing to Audio and Acoustics  (WASPAA 2025)}},
  ADDRESS = {Tahoe City, CA, United States},
  ORGANIZATION = {{IEEE}},
  YEAR = {2025},
}
```

## References

\[1\] R. C. Nongpiur. Impulse noise removal in speech using wavelets. In ICASSP, 2008.

\[2\] Z. Prša, N. Holighaus, and P. Balazs. Fast matching pursuit with multi‐Gabor dictionaries. ACM TOMS, 2021.