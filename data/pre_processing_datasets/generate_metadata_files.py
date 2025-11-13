import os
import argparse
import data_wrappers


def main(data_dir):
    # Generate metadata files for impulse datasets
    data_wrappers.ESC50(os.path.join(data_dir, 'esc50', 'ESC-50-master')).generate_metadata_file()
    data_wrappers.ReaLISED(os.path.join(data_dir, 'realised')).generate_metadata_file()
    data_wrappers.VocalSound(os.path.join(data_dir, 'vocalsound')).generate_metadata_file()
    data_wrappers.Nonspeech7k(os.path.join(data_dir, 'nonspeech7k')).generate_metadata_file()
    data_wrappers.FreesoundOneShotPercussive(os.path.join(data_dir, 'one-shot_percussive_sounds')).generate_metadata_file()

    # Generate metadata files for background datasets
    data_wrappers.Arte(os.path.join(data_dir, 'arte')).generate_metadata_file()
    data_wrappers.Cas2023(os.path.join(data_dir, 'cas2023')).generate_metadata_file()
    data_wrappers.CochlScene(os.path.join(data_dir, 'cochlscene')).generate_metadata_file()
    data_wrappers.Dcase2018(os.path.join(data_dir, 'dcase2018', 'TUT-urban-acoustic-scenes-2018-development')).generate_metadata_file()
    data_wrappers.VehicleInteriorSound(os.path.join(data_dir, 'visc_son')).generate_metadata_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Directory of downloaded datasets')
    args = parser.parse_args()
    main(args.data_dir)
