from pydub import AudioSegment
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from checking_impulse_sound import CheckingIfImpulse

import os
import shutil
import librosa
import numpy as np
import soundfile as sf

def remove_signal_edge_silence(sig, threshold=1e-7):
  """
  Remove silence at left and right edges of a signal

  Parameters
  ----------
  sig: numpy array with shape (n_channels, n_samples_ini)
     waveform of the input signal

  threshold: float (default 1e-7)
    threshold of the energy normalized cumulative sum used to detect
    silence durations

  Returns
  -------
  numpy array with shape (n_channels, n_samples_out)
     waveform of the input signal with start and end silences removed
  """

  energy = np.mean(np.square(sig), axis=0)

  left_cumsum = np.cumsum(energy)
  left_cumsum /= np.max(left_cumsum+1e-7)
  left_cut_idx = np.sum(left_cumsum < threshold)

  right_cumsum = np.cumsum(energy[::-1])
  right_cumsum /= np.max(right_cumsum+1e-7)
  right_cut_idx = len(energy) - np.sum(right_cumsum < threshold)

  #   plt.figure()
  #   plt.plot(energy/np.max(energy))
  #   plt.plot(left_cumsum, 'r')
  #   plt.plot(right_cumsum[::-1], 'k')
  #   plt.stem([left_cut_idx], [1.], 'r')
  #   plt.stem([right_cut_idx], [1.], 'k')

  return sig[:, left_cut_idx:right_cut_idx][0]

class RemovingNonImpulsive():
    """Class to create an impulse-free version of a dataset"""
    def __init__(self, original_root_dir, only_impulse_root_dir):
        self.original_root_dir = original_root_dir
        self.only_impulse_root_dir = only_impulse_root_dir
        self.failed_files = []

    @staticmethod
    def check_if_impulse(file_path, plot_figure: bool = False):
        """Detect impulses for a given audio file path

        Parameters
        ----------
        file_path : str
            Audio file path
        plot_figure : bool, optional
            Plotting utility, by default False

        Returns
        -------
        bool, list, list
            is_impulse: True or False
            impulse_times: list with impulse onsets
            merged_impulse_windows: list of lists with start and end time for each impulse
        """
        try:
            # Read audio file
            data, sr = librosa.load(file_path)  # Load with original sample rate
            
            # Convert to mono if needed
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)  # Averaging channels to convert to mono            

            # Commencer par retirer les edge silences ?
            data = remove_signal_edge_silence(data.reshape(1, -1), threshold=0.001*np.max(np.abs(data)))

            # Resample to 44100
            if sr != 44100:
                data = librosa.resample(data, orig_sr=sr, target_sr=44100)  # Resample
                sr = 44100

            impulse_module = CheckingIfImpulse(sr=44100)
            is_impulse = impulse_module.forward(signal=data, plot_figure=plot_figure)

            if plot_figure:
                plt.show()  # Ensure each plot is displayed before moving to the next iteration

            return is_impulse, data, sr
        except Exception as e:
            print(f'Error in file {file_path}: {e}')
            return None, None, None

    def replicate_folder_structure(self, file_extensions=('.wav', '.flac')):
        # Create destination directory if it doesn't exist
        if not os.path.exists(self.only_impulse_root_dir):
            os.makedirs(self.only_impulse_root_dir)
        else:
            pass
            # raise IsADirectoryError(f'{dest_dir} already exists!!!')

        # Walk through the source directory
        for root, _, files in os.walk(self.original_root_dir):
            # If audio_extensions is None or empty, include all files

            if file_extensions is None or not file_extensions:
                include_files = True
            else:
                include_files = any(file.endswith(tuple(file_extensions)) for file in files)

            if include_files:
                # Create directory structure in destination directory
                dest_path = os.path.join(self.only_impulse_root_dir,
                                        os.path.relpath(root, self.original_root_dir))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)


    def check_if_impulse_from_file_path(self, file_path):
        try:
            #print(f'Processing file: {file_path}')  # Debugging statement
            is_impulse, sig, sr = self.check_if_impulse(file_path, plot_figure=False)

            if not is_impulse:
                print(f'No impulses detected in file: {file_path}, not saving it')
                return

            if sig is None or sr is None:
                print(f'Failed to process audio file: {file_path}')
                return

            # Ensure dtype
            sig = sig.astype(np.float64)

            dst_file_path = file_path.replace(self.original_root_dir,
                                              self.only_impulse_root_dir)

            # Replace old copied file
            try:
                sf.write(dst_file_path, sig, sr)
                print(f'Successfully processed file: {file_path}')  # Debugging statement
            except Exception as e:
                print(f'Error writing file {dst_file_path}: {e}')
                self.failed_files.append(file_path)
        except Exception as e:
            print(f'File {file_path} failed! Error: {e}')
            self.failed_files.append(file_path)

    def generate_only_impulse_dataset(self):
        """Clean a dataset from impulses and store it in dst root dir

        Parameters
        ----------
        src_root_dir : str
            Original dataset directory
        dst_root_dir : str
            New, impulse-free dataset's directory
        """
        print("Replicating folder structure...")
        # Replicate src dir structure
        self.replicate_folder_structure(file_extensions=None)

        # Copy all non-audio files
        audio_exts = ['wav', 'mp3', 'flac', 'ogg', 'txt', 'md', 'json']
        audio_paths = []
        for root, _, files in os.walk(self.original_root_dir):
            if "_MACOSX" in root:
                continue
            for file in files:
                try:
                    file_ext = file.rsplit('.', 1)[1]
                except IndexError:
                    file_ext = ''
                if file_ext not in audio_exts:
                    src = os.path.join(os.path.join(self.original_root_dir, root, file))
                    dst = os.path.join(self.only_impulse_root_dir, root.split(self.original_root_dir)[1][1:], file)
                    shutil.copyfile(src, dst)
                else:
                    audio_paths.append(os.path.join(root, file))

        # Initialize the multiprocessing pool
        #set_start_method('spawn', force=True)
        print("Processing audios...")
        with Pool(cpu_count()-1) as pool:
            list(tqdm(pool.imap_unordered(self.check_if_impulse_from_file_path, audio_paths), total=len(audio_paths)))

