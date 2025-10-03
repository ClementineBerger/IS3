from pydub import AudioSegment
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from impulse_detection import ImpulseDetectionAlgorithm

import os
import shutil
import librosa
import numpy as np
import soundfile as sf

class ImpulseRemoval():
    """Class to create an impulse-free version of a dataset"""
    def __init__(self, original_root_dir, impulse_free_root_dir):
        self.original_root_dir = original_root_dir
        self.impulse_free_root_dir = impulse_free_root_dir
        self.failed_files = []

    @staticmethod
    def remove_segments_from_audio_to_array(src_file_path: str, segments_to_remove: list, crossfade_duration: int = 50):
        """
        Load an audio file, remove specified segments with crossfades, and
        return the result as a numpy array along with the sample rate.

        Parameters
        ----------
        src_file_path : str
            Path to the source audio file.
        segments_to_remove : list of tuples
            List of (start, end) tuples specifying the segments to remove
            in milliseconds.
        crossfade_duration : int, optional
            Duration of the crossfade in milliseconds, by default 50.

        Returns
        -------
        tuple
            A tuple containing:
            - numpy.ndarray: Processed audio as a numpy array.
            - int: Sample rate of the audio.
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_file(src_file_path)

            # Sort segments to remove by start time to process them in order
            segments_to_remove = sorted(segments_to_remove, key=lambda x: x[0])

            # Create a new audio segment to build the result
            processed_audio = AudioSegment.empty()

            # Keep track of the current position in the original audio
            current_position = 0

            for start, end in segments_to_remove:
                # Append the segment before the current segment to remove with crossfade
                if current_position < start:
                    segment_to_add = audio[current_position:start]
                    # Ensure crossfade duration does not exceed segment length
                    effective_crossfade_duration = min(
                        crossfade_duration, len(segment_to_add), len(processed_audio))
                    if processed_audio:
                        processed_audio = processed_audio.append(
                            segment_to_add, crossfade=effective_crossfade_duration)
                    else:
                        processed_audio = segment_to_add
                # Move the current position to the end of the segment to remove
                current_position = end

            # Append the remaining part of the audio after the last segment to
            # remove with crossfade
            if current_position < len(audio):
                segment_to_add = audio[current_position:]
                effective_crossfade_duration = min(
                    crossfade_duration, len(segment_to_add), len(processed_audio))
                if processed_audio:
                    processed_audio = processed_audio.append(
                        segment_to_add, crossfade=effective_crossfade_duration)
                else:
                    processed_audio = segment_to_add

            # Convert the processed audio to a numpy array
            new_sig = np.array(processed_audio.get_array_of_samples())

            # If the audio is stereo, reshape the array
            if processed_audio.channels == 2:
                new_sig = new_sig.reshape((-1, 2))

            # Get the sample rate
            sample_rate = processed_audio.frame_rate

            return new_sig, sample_rate
        except Exception as e:
            print(f'Error processing audio file {src_file_path}: {e}')
            return None, None

    @staticmethod
    def detect_impulses(file_path, plot_figure: bool = False):
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

            # Resample to 16000
            if sr != 16000:
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)  # Resample

            impulse_module = ImpulseDetectionAlgorithm(onset_selection_window_size_s=10., sr=16000)
            is_impulse, impulse_times, merged_impulse_windows = \
                impulse_module.impulse_detection(data, plot_figure=plot_figure)

            if plot_figure:
                plt.show()  # Ensure each plot is displayed before moving to the next iteration

            return is_impulse, impulse_times, merged_impulse_windows
        except Exception as e:
            print(f'Error detecting impulses in file {file_path}: {e}')
            return None, None, None

    def replicate_folder_structure(self, file_extensions=('.wav', '.flac')):
        # Create destination directory if it doesn't exist
        if not os.path.exists(self.impulse_free_root_dir):
            os.makedirs(self.impulse_free_root_dir)
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
                dest_path = os.path.join(self.impulse_free_root_dir,
                                         os.path.relpath(root, self.original_root_dir))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

    def remove_impulses_from_file_path(self, file_path):
        try:
            #print(f'Processing file: {file_path}')  # Debugging statement
            _, _, merged_impulse_windows = self.detect_impulses(file_path, plot_figure=False)

            if merged_impulse_windows is None:
                print(f'No impulses detected in file: {file_path}')
                return

            # Convert from list of lists in secs to list of tuples in msecs
            segments_to_remove = []
            for sublist in merged_impulse_windows:
                segments_to_remove.append((sublist[0] * 1000, sublist[1] * 1000))

            sig, sr = self.remove_segments_from_audio_to_array(file_path, segments_to_remove)

            if sig is None or sr is None:
                print(f'Failed to process audio file: {file_path}')
                return

            # Ensure dtype
            sig = sig.astype(np.float64)

            dst_file_path = file_path.replace(self.original_root_dir,
                                              self.impulse_free_root_dir)

            # Replace old copied file
            try:
                sf.write(dst_file_path, sig, sr)
                #print(f'Successfully processed file: {file_path}')  # Debugging statement
            except Exception as e:
                print(f'Error writing file {dst_file_path}: {e}')
                self.failed_files.append(file_path)
        except Exception as e:
            print(f'File {file_path} failed! Error: {e}')
            self.failed_files.append(file_path)

    def generate_impulse_free_dataset(self):
        """Clean a dataset from impulses and store it in dst root dir

        Parameters
        ----------
        src_root_dir : str
            Original dataset directory
        dst_root_dir : str
            New, impulse-free dataset's directory
        """

        # Replicate src dir structure
        self.replicate_folder_structure(file_extensions=None)

        # Copy all non-audio files
        audio_exts = ['wav', 'mp3', 'flac', 'ogg']
        audio_paths = []
        for root, _, files in os.walk(self.original_root_dir):
            for file in files:
                try:
                    file_ext = file.rsplit('.', 1)[1]
                except IndexError:
                    file_ext = ''
                if file_ext not in audio_exts:
                    src = os.path.join(os.path.join(self.original_root_dir, root, file))
                    dst = os.path.join(self.impulse_free_root_dir, root.split(self.original_root_dir)[1][1:], file)
                    shutil.copyfile(src, dst)
                else:
                    audio_paths.append(os.path.join(root, file))

        # Initialize the multiprocessing pool
        #set_start_method('spawn', force=True)
        with Pool(cpu_count()-1) as pool:
            list(tqdm(pool.imap_unordered(self.remove_impulses_from_file_path, audio_paths), total=len(audio_paths)))

