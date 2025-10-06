# Adding a New Dataset

The current data loading relies on the following parts:

- dataset mapping to SALT taxonomy -> serves the use of common labels set

- dataset loader

- dataset loader wrapper in the [data_loader.py](./data_loader.py) -> Used to include the dataset in the corresponding loader (Backgrounds/Impulses)


## Mapping to SALT

To add a new dataset, you start by mapping it in the corresponding taxonomy (csv files).

For events/impulses: `./assets/impulses_event_mapping.tsv`

For scenes/backgrounds: `./assets/audible_scene_mapping.tsv`

Here is an example of datasets labels refer to a human sneezing mapped into the standard label "sneezing".

| standard_event | dataset_label | dataset |
|---------------|--------------|--------------|
| sneezing      | sneeze       | Nonspeech7k  |
| sneezing      | sneeze       | VocalSound   |
| sneezing      | sneezing     | ESC50        |


Adding a new dataset may come to 2 possible situations:

**1. Mapping the new dataset's label into an existing standard event**

This is the case when one of the standard labels of the taxonomy corresponds to the new label to be mapped.

| standard_event | dataset_label | dataset |
|---------------|--------------|--------------|
| sneezing      | sneeze       | Nonspeech7k  |
| sneezing      | sneeze       | VocalSound   |
| sneezing      | sneezing     | ESC50        |
| sneezing      | Sneeze_NEW   | NEW_DATASET  |

**2. Mapping the new dataset's label into a new standard event** 

This is the case when a new dataset's label does not fit to any of the standard labels. Then a new standard label is created.

| standard_event | dataset_label | dataset |
|---------------|--------------|--------------|
| sneezing      | sneeze       | Nonspeech7k  |
| sneezing      | sneeze       | VocalSound   |
| sneezing      | sneezing     | ESC50        |
|               |              |              |
| NEW_STD_LABEL | NEW_DB_LABEL | NEW_DATASET  |


Once the dataset is mapped, the taxonomy's roots dictionary should be updated:

```python
from utils import SaltWrapper

# Initialize the mapper
salt_mapper = SaltWrapper()

# Update roots dict for the evets (impulses)
salt_mapper.event_mapper.generate_taxonomy_roots()

# Update roots dict for the scenes (backgrounds)
salt_mapper.scene_mapper.generate_taxonomy_roots()
```


## Dataset Wrapper in the data_loader.py

Next, a wrapper class should be placed in the [data_loader.py](./data_loader.py) module as the example below:

```python
class Esc50Wrapper(DatasetLoader):
  def __init__(self):
    super().__init__(
      root_dir=os.path.join(config.AUDIBLE_DCASE_DATA, 'ESC-50_v2_no_impulses'),
      module=event,
      cls='ESC50',
      label_col='label',
      cols_to_keep=COLS_TO_KEEP,
      impulse_type='natural'
    )
```

Fields:

- **root_dir**: The root directory of the dataset
- **module**: event or scene depending on the dataset type
- **cls**: The class' name of the data loader in the module
- **label_col**: the name of the column which contains labels in the pdf_metadata field of the data loader
- **cols_to_keep**: Used to remove unecessary columns
- **impulse_type**: "natural" for real data, "synthetic" for synthetic data. None when the dataset has no impulses (background data)

Once the Wrapper class is ready, it should be registered in one of the two types of data loaders (background/Foreground):

```python
DatasetLoaderRegistry.register_background(Esc50Wrapper)
```

See more examples in the [data_loader.py](./data_loader.py)