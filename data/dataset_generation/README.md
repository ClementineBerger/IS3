# Dataset generation

## Overview

## py-salt

The generation process relies on the taxonomy *SALT: Standardized Audio Scene Taxonomy* \[3\] to map and unify the labels of the various datasets to a chosen set of labels. This helps managing the diversity of the datasets and generating a balanced dataset.

You will need to install the `py-salt` library (https://github.com/tpt-adasp/salt) by cloning it and follow the procedure described in `add_new_dataset_tutorial.md`. The `.tsv` files in the `assets` are provided as examples and can be modified to include different datasets.

The script `scene_mapping.py` is a modified version of the `py-salt` `event_mapping.py` to handle scene mapping. This should be added to the `py-salt` package under `salt/py-salt/py_salt/scene_mapping.py`.

## Data loading

For each of the datasets used, you will need to create a dataset loader. Dataset loaders used in this work are based on a private package so you will need to create your own dataset loaders. Some examples of dataset loaders are provided in the `data_loaders.py`.

## Configuration

The configuration file `config.py` contains the paths to the datasets, the assets, and the saving directory. You will need to modify the paths to match your local setup. It contains also the parameters for the generation process and the data augmentation.