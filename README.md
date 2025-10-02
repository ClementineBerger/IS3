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


## Cite us !

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
