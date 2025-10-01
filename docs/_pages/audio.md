---
title: "Audio Examples"
layout: single
permalink: /audio
author_profile: false
header:
  overlay_color: "#f29c3f" #"#000"
  overlay_filter: "0.3"
excerpt: #"title"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---


<!-- <br/> -->

# Synthetic examples on the test set

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
                <th><center>Stationary Background</center></th>
                <th><center>Impulsive sounds</center></th>        
                <th><center>Mix</center></th>   
            </tr>               
        </thread>
        <tbody>
            <tr>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/bkg.wav" type="audio/wav"/>
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <audio controls controlslist="nodownload">
                        <source src="audio/example1/impulse.wav" type="audio/wav"/>
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
    <table>
        <thead>
            <tr>
                <th><center></center></th>
                <th><center>Impulsive sounds</center></th>
                <th><center>Stationary Background</center></th>
            </tr>               
        </thead>
        <tbody>
            <tr>
                <td>HPSS $p_m = 1$</td>
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
        </tbody>
    </table>
</body>
</html>
<br/>