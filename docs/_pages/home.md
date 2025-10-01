---
title: ""
layout: single
permalink: /
author_profile: false
classes: wide
header:
  overlay_color: "#cc3131" #"#000"
  overlay_filter: "0.3"
excerpt: #"title"
---

<!-- You will find audio examples [here](./audio.md) and additional results [here](./results.md). -->

# Abstract

We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. 
Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics.
{: .text-justify}

To this end, we introduce IS³, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics.
{: .text-justify}

**Index Terms** - source separation, deep filtering, impulsive sounds