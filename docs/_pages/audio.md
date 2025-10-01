---
title: "Audio Examples"
layout: single
permalink: /audio
author_profile: false
header:
  overlay_color: "#cc3131" #"#000"
  overlay_filter: "0.3"
excerpt: #"title"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---


This is the audio example page. You can find some audio examples of ambient noise in several acoustic scenes and music processed by different systems :
- Baseline model by Estreder et al.
- Proposed model DPNMM without any power constraint
- DPNMM with $\Delta \mathcal{P}_{max} = $ 2, 1, 0.5 dBA
{: .text-justify}

For each environment, the noise is filtered with a headphone's frequency response to simulate its passive attenuation. 
{: .text-justify}

You can listen to the unprocessed music, the noise and the mix of the two. Then you can compare the perception of the noise in the unprocessed mix with the mixes obtained with the different systems.
{: .text-justify}

<br/>

# Train / subway 

> Unprocessed audios

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th><center>Unprocessed music</center></th>
                <th><center>Noise</center></th>        
                <th><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_init_music.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_noise.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_init_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Processed music 

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>Estreder</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_estreder_001_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (1 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_neural_016_power_1_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

> Additional configurations

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>DPNMM (no power)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_neural_016_nopower_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (2 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_neural_016_power_2_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (0.5 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/train/train_neural_016_power_0-5_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

# Restaurant / Café

> Unprocessed audios

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th><center>Unprocessed music</center></th>
                <th><center>Noise</center></th>        
                <th><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls>
                        <source src="audio/restaurant/restaurant_init_music.wav" type="audio/wav"/>
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_noise.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_init_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Processed music

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>Estreder</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_estreder_001_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (1 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_neural_016_power_1_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

> Additional configurations

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>DPNMM (no power)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_neural_016_nopower_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (2 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_neural_016_power_2_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (0.5 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/restaurant/restaurant_neural_016_power_0-5_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

# Street, urban area

> Unprocessed audios

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th><center>Unprocessed music</center></th>
                <th><center>Noise</center></th>        
                <th><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_init_music.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_noise.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_init_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Processed music

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>Estreder</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_estreder_001_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (1 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_neural_016_power_1_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

> Additional configurations

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>DPNMM (no power)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_neural_016_nopower_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (2 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_neural_016_power_2_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (0.5 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/urban/urban_neural_016_power_0-5_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

# Inside a bus 

> Unprocessed audios

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <thread>
            <tr>
                <th><center>Unprocessed music</center></th>
                <th><center>Noise</center></th>        
                <th><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_init_music.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_noise.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_init_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>                   
        </tbody>
    </table>
</body>
</html>
<br/>

> Processed music

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>Estreder</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_estreder_001_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (1 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_neural_016_power_1_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>

> Additional configurations

<html>
<!-- <head>
    <title>Tableau d'Audios</title>
</head> -->
<body>
    <table>
        <!-- <thead>
            <tr>
                <th><center>Title</center></th>
                <th><center>Audio</center></th>
            </tr>               
        </thead> -->
        <tbody>
            <tr>
                <td>DPNMM (no power)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_neural_016_nopower_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (2 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_neural_016_power_2_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
            <tr>
                <td>DPNMM (0.5 dBA)</td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/bus/bus_neural_016_power_0-5_mix.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
<br/>