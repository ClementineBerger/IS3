---
title: "Audio Examples"
layout: single
permalink: /audio
author_profile: false
header:
  overlay_color: "#eb9f34" #"#000"
  overlay_filter: "0.1"
excerpt: #"title"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

You will find here audio examples of the different separation methods presented in the paper below, both on synthetic mixtures from the test set and on real-life recordings. 
{: .text-justify}

We compare our proposed IS³ model with the Harmonic-Percussive Sound Separation (HPSS) method, adapted from music signal processing research, using two different margin parameters ($p_m = 1$ and $p_m = 2$), the wavelet-based method from Nongpiur et al. \[1\] and a Conv-TasNet model \[2\] trained on the same data as IS³.
{: .text-justify}

> **References**
> \[1\] R. C. Nongpiur. Impulse noise removal in speech using wavelets. In ICASSP, 2008.
> \[2\] Y. Luo and N. Mesgarani. Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.
{: .text-justify}

<!-- <br/> -->

# Synthetic examples on the test set

These initial examples are derived from the test set, generated using the same pipeline as the training and validation datasets described in the paper. 
{: .text-justify}

We select an example from the test set, which consists of the following components:
- A stationary background track
- An impulsive sounds track
- A mixture track

The mixture track serves as the input for the four separation methods to extract the stationary and impulsive components.
{: .text-justify}

## Example 1

> Clean signals and mix

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Impulsive sounds</center></th>        
                <th style="font-family: 'Montserrat', sans-serif;"><center>Stationary Background</center></th>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Separated signals with the different methods

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table style="font-family: 'Montserrat', sans-serif; font-weight: 400; text-align: center;">
        <thread>
            <tr>
                <th><center></center></th>
                <th style="font-family: 'Montserrat', sans-serif;">Impulsive sounds</th>
                <th style="font-family: 'Montserrat', sans-serif;">Stationary Background</th>
        </tr>            
        </thread>
        <tbody>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">IS³</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/is3_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/is3_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>        
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 1$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/hpss_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/hpss_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 2$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/hpss2_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/hpss2_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Nongpiur</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/wavelet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/wavelet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Conv-TasNet</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/convtasnet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/convtasnet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

## Example 2

> Clean signals and mix

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Impulsive sounds</center></th>        
                <th style="font-family: 'Montserrat', sans-serif;"><center>Stationary Background</center></th>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Separated signals with the different methods

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table style="font-family: 'Montserrat', sans-serif; font-weight: 400; text-align: center;">
        <thread>
            <tr>
                <th><center></center></th>
                <th style="font-family: 'Montserrat', sans-serif;">Impulsive sounds</th>
                <th style="font-family: 'Montserrat', sans-serif;">Stationary Background</th>
        </tr>            
        </thread>
        <tbody>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">IS³</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/is3_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/is3_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>        
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 1$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/hpss_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/hpss_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 2$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/hpss2_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/hpss2_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Nongpiur</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/wavelet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/wavelet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Conv-TasNet</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/convtasnet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example2/convtasnet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

## Example 3 : empty impulsive track 

> Clean signals and mix

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Impulsive sounds</center></th>        
                <th style="font-family: 'Montserrat', sans-serif;"><center>Stationary Background</center></th>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Separated signals with the different methods

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table style="font-family: 'Montserrat', sans-serif; font-weight: 400; text-align: center;">
        <thread>
            <tr>
                <th><center></center></th>
                <th style="font-family: 'Montserrat', sans-serif;">Impulsive sounds</th>
                <th style="font-family: 'Montserrat', sans-serif;">Stationary Background</th>
        </tr>            
        </thread>
        <tbody>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">IS³</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/is3_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/is3_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>        
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 1$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/hpss_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/hpss_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 2$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/hpss2_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/hpss2_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Nongpiur</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/wavelet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/wavelet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Conv-TasNet</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/convtasnet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example3/convtasnet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>


**Observations:** The HPSS methods exhibit significant leakage of both stationary and impulsive components in the separated tracks. Increasing the margin parameter ($p_m = 2$) reduces the leakage of the stationary background into the impulsive track.  
{: .text-justify}

Nongpiur's method performs poorly on the impulsive track, introducing audio artefacts. On the stationary track, the original method from Nongpiur’s article only attenuates impulsive sounds, which remain partially present. It is worth noting that the parameter selection in this wavelet-based approach is highly dependent on the type of impulses and ambient sounds (speech in the original article). While efforts were made to optimize parameters for our context, the diversity of sound types and acoustic scenes in our study leads to inconsistent performance across examples.
{: .text-justify}

Conv-TasNet surpasses other baselines but still shows some leakage of the stationary background into the impulsive track. In contrast, IS³ delivers the best separation, producing a clean impulsive track and a clean stationary track. However, there is a slight attenuation of the resonance in impulsive sounds, making them sound slightly drier compared to the target track. Additionally, in examples where the impulsive track's frequency components are more dispersed, the separation process becomes more challenging, leading to a slight reduction in the background track's sound level during impulse events.
{: .text-justify}

Notably, in the third example where the impulsive track is silent, IS³ successfully generates a completely silent impulsive track, whereas other methods introduce artefacts or stationary background leakage.
{: .text-justify}


# Real recordings examples 

To better appreciate the performance of the model, it is also interesting to test it on real-life recordings (for which we don't have groundtruth for either the impulsive or the stationary sources).
{: .text-justify}

## Example 1

> Original recording

<html>
<body>
    <table>
        <thread>
            <tr>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/real_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Separated signals with the different methods

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table style="font-family: 'Montserrat', sans-serif; font-weight: 400; text-align: center;">
        <thread>
            <tr>
                <th><center></center></th>
                <th style="font-family: 'Montserrat', sans-serif;">Impulsive sounds</th>
                <th style="font-family: 'Montserrat', sans-serif;">Stationary Background</th>
        </tr>            
        </thread>
        <tbody>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">IS³</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/is3_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/is3_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>        
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 1$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/hpss_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/hpss_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 2$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/hpss2_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/hpss2_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Nongpiur</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/wavelet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/wavelet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Conv-TasNet</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/convtasnet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example4/convtasnet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>


## Example 2

> Original recording

<html>
<body>
    <table>
        <thread>
            <tr>
                <th style="font-family: 'Montserrat', sans-serif;"><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/real_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Separated signals with the different methods

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table style="font-family: 'Montserrat', sans-serif; font-weight: 400; text-align: center;">
        <thread>
            <tr>
                <th><center></center></th>
                <th style="font-family: 'Montserrat', sans-serif;">Impulsive sounds</th>
                <th style="font-family: 'Montserrat', sans-serif;">Stationary Background</th>
        </tr>            
        </thread>
        <tbody>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">IS³</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/is3_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/is3_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>        
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 1$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/hpss_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/hpss_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">HPSS $p_m = 2$</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/hpss2_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/hpss2_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Nongpiur</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/wavelet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/wavelet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td style="font-family: 'Montserrat', sans-serif;">Conv-TasNet</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/convtasnet_impulse.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example5/convtasnet_bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>