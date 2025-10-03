from utils import ImpulseRemoval
from multiprocessing import set_start_method
import os
import pandas as pd


def main():
    src_root_dir = "path_to_your_dataset"
    dst_root_dir = "path_to_your_impulse_free_dataset"
    LOG_FILE_PATH = "path_to_log_failed_files.csv"

    assert os.path.isdir(src_root_dir)

    # Your logic for generating the impulse-free dataset goes here
    i = ImpulseRemoval(original_root_dir=src_root_dir,
                    impulse_free_root_dir=dst_root_dir)

    i.generate_impulse_free_dataset()

    if len(i.failed_files) != 0:
        pd.DataFrame({'file_path': i.failed_files}).to_csv(LOG_FILE_PATH)


if __name__ == '__main__':
    # Set the start method for multiprocessing to 'spawn'
    set_start_method('spawn', force=True)
    
    # Call the main function to run the process
    main()
