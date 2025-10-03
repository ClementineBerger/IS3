# Impulse Sound Removal

Pipeline to remove impulses from audio files.

## Requirements

We are relying on the Matlab/Octave toolbox ( https://ltfat.org/ or github https://github.com/ltfat) running in the python script using Oct2Py ( https://pypi.org/project/oct2py/ ).

### Installing Octave

Follow the steps described here : https://wiki.octave.org/Octave_for_Debian_systems .

### In VSCode
In VSCode you can install the Octave extension.

Verify that you find in you preferences json file :
```
"terminal.integrated.env.windows": {
    "PATH": "/use/bin/octave"
},
```

Then, you can open an octave prompt in the terminal using `octave` to use the `pkg` command.

### Installing LTFAT
Follow the steps described here (GNU Octave part): https://github.com/ltfat/ltfat/releases/tag/v2.6.0 . 

### Python Env 
To reproduce this study a [docker folder](./docker/) contains all the necessary tools to build a ready to use env.

This module uses the LTFAT library written in C 

## impulse_detection.py script

The algorithm is implemented in the ImpulseDetectionClass. 


Use :

```
from impulse_detection import ImpulseDetectionALgorithm

import librosa

sr = 16000
window_size = [512, 1024, 2048, 4096, 8192]  # window sizes
hop_size = np.array(window_size) / 4  # time shifts

signal, _ = librosa.load("path/to/audio.wav", sr=sr) 

impulse_module = ImpulseDetectionAlgorithm(
    sr=sr, 
    window_sizes = window_size, 
    hop_sizes = hop_sizes
)  # default values can be used for the other parameters

find_impulse, impulse_times, impulse_windows = impulse_model.impulse_detection(
    signal=signal,
    plot_figure=False
) # you can set plot_figure to True if you want to visualize the algorithm
```

