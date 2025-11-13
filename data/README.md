

- **pre_processing_datasets** Code to pre-process publicly available datasets to:
    - Detect and remove impulses from datasets used for backgrounds
    - Clean impulses from background noise from datasets used for foreground

- **synthetic_impulse_sounds**: Code to generate synthetic impulse sounds

- **synthetic_stationary_backgrounds**: Code to generate synthetic stationary backgrounds

- **dataset_generation**: Code to generate the dataset for IS3.




### Datasets:

Datasets used for impulses:
- [ESC-50](https://github.com/karolpiczak/ESC-50) (1.4G)
- [ReaLISED](https://zenodo.org/records/6488321) (786M)
- [VocalSound](https://github.com/YuanGongND/vocalsound) (18G)
- [NonSpeech7k](https://zenodo.org/records/10616533) (4.6G)
- [Freesound One-Shot Percussive Sounds](https://zenodo.org/records/4687854) (1G)

Datasets used for backgrounds:

- [ARTE](https://zenodo.org/records/2261633) (13G)
- [CAS 2023](https://zenodo.org/records/10616533) (13G)
- [CochlScene](https://zenodo.org/records/7080122) (98G)
- [Dcase2018 Task 1 Dataset](https://dcase.community/challenge2018/task-acoustic-scene-classification) (32G)
- [Vehicle Interior Sound Dataset](https://zenodo.org/records/5606504) (3.8G)
- Litis Rouen

We provide scripts to download the public datasets used for backgrounds and impulses. To download a dataset use the corresponding script specifying the output directory. E.g. for ESC-50:

```bash
# cd into download scripts 
cd PATH_TO_IS3/data/download_scripts

# run the script to download ESC-50
chmod +x download_esc50.sh && \
./download_esc50.sh /PATH/TO/ESC-50
```

For reproducability purposes, we contain samples of each dataset to showcase each pre-processing step.
