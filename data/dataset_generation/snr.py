import numpy as np
import librosa
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve
import pyloudnorm as pyln
from scipy.optimize import minimize


def compute_mel_spectrogram(sig_in: np.array,
                            sampling_rate: int = 44100,
                            win_len: int = 2048,
                            hop_len: int = 512,
                            power: float = 2.0,
                            n_mels: int = 128,
                            fmax: int = 11000):
  """
  Mel spectrogram extraction for standardized SNR value computations.
  The defaults values have been found to work well.

  Parameters
  ----------
  sig_in : np.array
      Input signal of shape (n_channels, n_samples).
  sampling_rate : int, optional
      Sampling rate of the input signal, by default 44100.
  win_len : int, optional
      Length of the window used for the STFT, by default 2048.
  hop_len : int, optional
      Number of samples between successive frames, by default 512.
  power : float, optional
      Exponent for the magnitude of the STFT, by default 2.0.
  n_mels : int, optional
      Number of mel bands to use, by default 128.
  fmax : int, optional
      Maximum frequency to include in the mel spectrogram, by default 11000.

  Returns
  -------
  np.array
      Mel spectrogram of shape (n_freq, n_time_frames, n_channels).
  """

  if np.any(np.isnan(sig_in)):
    raise ValueError('The array contains NaN values.')
  if np.any(np.isinf(sig_in)):
    raise ValueError('The array contains infinite values.')

  # Initialize empty list to store spectrograms
  spectro = []

  # Adjust window length and hop length if sampling rate is not 44100
  if sampling_rate != 44100:
    win_len = int((float(win_len) / 44100) * sampling_rate)
    hop_len = int((float(hop_len) / 44100) * sampling_rate)

    # Adjust number of mel bands if sampling rate is less than 22000
    if sampling_rate < 22000:
      mel_f = librosa.core.mel_frequencies(n_mels + 2, fmin=0, fmax=fmax)
      fmax = sampling_rate / 2
      n_mels = np.max(np.where(mel_f < fmax)) - 1

  # Compute mel spectrogram for each channel of input signal
  for chan_idx in range(sig_in.shape[0]):
    spectro.append(librosa.feature.melspectrogram(y=sig_in[chan_idx, :],
                                                  sr=sampling_rate,
                                                  n_fft=win_len,
                                                  hop_length=hop_len,
                                                  power=power,
                                                  n_mels=n_mels,
                                                  fmax=fmax))

  # Stack spectrograms along last axis to produce 3D array
  return np.stack(spectro, axis=-1)


def compute_tf_mask_significant_energy(sig_spectrogram,
                                       energy_threshold=0.999,
                                       display_mask=False):
  """
  Compute a binary mask from a signal spectrogram which preserves
  the tf-points that contribute to 100*energy_threshold % of the total
  energy

  Parameters
  ----------
  sig_spectrogram: ndarray
      power (i.e squared magnitude) of a time-frequency representation

  energy_threshold: float (default 0.999)
      parameter value should be in [0,1]
      amount of total energy that should be preserved by the
      time-frequency mask

  display_mask: bool (default False)
      display the mask computed

  Returns
  -------
      a numpy array with the same shape as sig_spectrogram
      corresponding to a time-frequency mask
  -------
  """

  vect_spectro = np.reshape(
      sig_spectrogram, -1) / np.sum(sig_spectrogram + np.finfo(float).eps)

  sort_indices = np.argsort(vect_spectro)[::-1]
  n_tf_point_with_significant_energy = np.sum(np.cumsum(
      vect_spectro[sort_indices]) < energy_threshold)

  tf_mask = np.zeros(sig_spectrogram.size, dtype=int)
  tf_mask[sort_indices[0:n_tf_point_with_significant_energy]] = 1
  tf_mask = np.reshape(tf_mask, sig_spectrogram.shape)

  if display_mask:

    n_channels = sig_spectrogram.shape[-1]
    plt.figure()
    for chan_idx in range(n_channels):
      plt.subplot(2, n_channels, chan_idx + 1)
      plt.imshow(np.log(sig_spectrogram[0, :, :, chan_idx]),
                  aspect='auto', origin='lower', cmap='jet')
      plt.subplot(2, n_channels, n_channels + chan_idx + 1)
      plt.imshow(tf_mask[0, :, :, chan_idx],
                  aspect='auto', origin='lower', cmap='jet')
    plt.show()

  return tf_mask


def compute_background_gain_for_balanced_mix(reference_signal: np.array,
                                             background_signal: np.array,
                                             sampling_rate: int,
                                             display: bool = False):
  """
  Compute a global gain for a background signal such that the sum of the
  reference and background signals have the same perceptual strength.
  Based on the computation of a median time-frequency SNRs on time-frequency
  regions containing significant energy

  Parameters
  ----------
  reference_signal:  numpy array with shape (n_channels, n_samples)
      waveform of the reference signal

  background_signal: numpy array with shape (n_channels, n_samples)
      waveform of the background signal

  sampling_rate: int
      sampling_frequency of both signals

  display: bool (default False)
      For debugging.

  Return
  ------
  background_gain: float (scalar)
      gain to apply to the background signal (background_signal *= gain) in
      order to obtain a balanced mix between the reference and background
      signals
  """
  # Check if the shapes of the reference and background signals are the same
  if reference_signal.shape != background_signal.shape:
    raise ValueError('reference and background signal '
                     'should be of same shape')

  # Compute Mel spectrograms for the reference and background signals

  reference_spectro = compute_mel_spectrogram(reference_signal, sampling_rate)
  background_spectro = compute_mel_spectrogram(background_signal, sampling_rate)

  # Compute binary time-frequency masks defining time-frequency regions
  # where the two signals present significant energy
  reference_mask = compute_tf_mask_significant_energy(reference_spectro,
                                                      energy_threshold=0.999,
                                                      display_mask=False)

  background_mask = compute_tf_mask_significant_energy(background_spectro,
                                                       energy_threshold=0.999,
                                                       display_mask=False)

  # Compute time-frequency SNRs on these time-frequency regions
  tf_snr = np.divide(reference_spectro, background_spectro + np.finfo(float).eps)
  tf_snr = np.multiply(np.multiply(reference_mask, background_mask), tf_snr)

  # Reshape the SNR array and convert it to dB scale
  snr = np.reshape(tf_snr, -1)
  snr = 10.0 * np.log10(snr[snr > 0])

  # Compute gain that produces a median 0dB SNR on these time-frequency regions
  if len(snr) == 0:
    background_gain = 1.
  else:
    background_gain = math.sqrt(10**(np.median(snr) / 10.))

  # If display is True, plot the Mel spectrograms and SNR map for debugging
  if display:
    plt.figure()
    plt.subplot(311)
    plt.imshow(10 * np.log10(np.multiply(reference_spectro, reference_mask)[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(10 * np.log10(np.multiply(background_spectro, background_mask)[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(10 * np.log10(tf_snr[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()

    plt.show()

  return background_gain


def compute_snr(output_signal,
                input_signal,
                sampling_rate,
                display=False):
  """
  Computation of a median time-frequency SNRs on time-frequency
  regions containing significant energy

  Parameters
  ----------
  output_signal:  numpy array with shape (n_channels, n_samples)
      waveform of the output signal

  input_signal: numpy array with shape (n_channels, n_samples)
      waveform of the input signal

  sampling_rate: int
      sampling_frequency of both signals

  display: bool (default False)
      For debugging.

  Return
  ------
  snr : median snr over the mel spectrograms
  """

  if output_signal.shape != input_signal.shape:
    raise ValueError('reference and background signal '
                     'should be of same shape')

  # -- Compute a binary time-frequency mask defining tf regions
  # where the two signals present significant energy

  output_spectro = compute_mel_spectrogram(
      output_signal, sampling_rate)
  output_mask = compute_tf_mask_significant_energy(
      output_spectro, energy_threshold=0.999, display_mask=False)

  input_spectro = compute_mel_spectrogram(
      input_signal, sampling_rate)
  input_mask = compute_tf_mask_significant_energy(
      input_spectro, energy_threshold=0.999, display_mask=False)

  # -- Compute tf-wise SNRs on these tf regions

  tf_snr = np.divide(output_spectro,
                     input_spectro + np.finfo(float).eps)

  tf_snr = np.multiply(np.multiply(input_mask, output_mask), tf_snr)

  snr = np.reshape(tf_snr, -1)

  if display:

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(311)
    plt.title('Output spectro')
    plt.imshow(
        10 *
        np.log10(
            output_spectro[
                :,
                :,
                0] +
            1e-6),
        aspect='auto',
        origin='lower',
        cmap='jet', vmin=-60, vmax=10)
    plt.colorbar()
    plt.subplot(312)
    plt.title('Input spectro')
    plt.imshow(
        10 *
        np.log10(
            input_spectro[
                :,
                :,
                0] +
            1e-6),
        aspect='auto',
        origin='lower',
        cmap='jet', vmin=-60, vmax=10)
    plt.colorbar()
    plt.subplot(313)
    plt.title('SNR')
    plt.imshow(10 * np.log10(tf_snr[:, :, 0] + 1e-6),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

  return 10 * np.log10(np.median(snr[snr > 0]))


def add_db(reference_signal: np.array, db: float):
  """
  Reduce the amplitude of a reference signal by a specified number of decibels.

  Parameters
  ----------
  reference_signal:  numpy array with shape (n_channels, n_samples)
      waveform of the reference signal

  db: float
      amount to enhance/reduce the amplitude of the reference signal,
      in decibels

  Return
  ------
  numpy array with shape (n_channels, n_samples)
      waveform of the reference signal with reduced amplitude
  """

  # Convert the decibel reduction to a linear scale factor
  gain_factor = 10**(db / 20)

  # Change the amplitude of the reference signal
  return reference_signal * gain_factor


def compute_power_based_snr(ref_signal, background_signals, target_snr):
  """Compute gains for background signals such that their combined power
  matches the reference signal, scaled logarithmically based on a given
  SNR value.

  Parameters
  ----------
  ref_signal : np.ndarray (n_channels, n_samples)
      The reference signal
  background_signals : list(np.ndarray (n_channels, n_samples))
      The background signals
  target_snr : float
      The SNR value between the power of the reference signal and the
      power of the combined background signals. 

  Returns
  -------
  _type_
      _description_
  """
  ref_signal = ref_signal.flatten()  # Ensure 1D array
  background_signals = [bg.flatten() for bg in background_signals]

  # Compute power of the reference signal
  p_ref = np.mean(ref_signal ** 2)

  # Compute power of each background signal
  p_bgs = np.array([np.mean(bg ** 2) for bg in background_signals])

  num_backgrounds = len(background_signals)

  # Compute gains for each background signal based on SNR
  if target_snr > 0:
    # Positive SNR: P_ref = P_combined * SNR
    gains = np.sqrt(p_ref / (target_snr * num_backgrounds * p_bgs))
  elif target_snr < 0:
    # Negative SNR: P_ref = P_combined / |SNR|
    gains = np.sqrt(p_ref / ((num_backgrounds / (abs(target_snr)) * p_bgs)))
  else:
    # Zero SNR: P_ref = P_combined
    gains = np.sqrt(p_ref / (num_backgrounds * p_bgs))

  return gains


def compute_background_gain_for_unbalanced_mix(reference_signal: np.array,
                                               background_signal: np.array,
                                               sampling_rate: int,
                                               target_snr: float,
                                               only_mask_with_reference: bool = False,
                                               only_mask_with_background: bool = False,                                               
                                               display: bool = False):
  """
  Compute a global gain for a background signal such that the sum of the
  reference and background signals have the same perceptual strength.
  Based on the computation of a median time-frequency SNRs on time-frequency
  regions containing significant energy

  Parameters
  ----------
  reference_signal:  numpy array with shape (n_channels, n_samples)
      waveform of the reference signal

  background_signal: numpy array with shape (n_channels, n_samples)
      waveform of the background signal

  sampling_rate: int
      sampling_frequency of both signals

  display: bool (default False)
      For debugging.

  Return
  ------
  background_gain: float (scalar)
      gain to apply to the background signal (background_signal *= gain) in
      order to obtain a balanced mix between the reference and background
      signals
  """
  # Check if the shapes of the reference and background signals are the same
  if reference_signal.shape != background_signal.shape:
    raise ValueError('reference and background signal '
                     'should be of same shape')

  # Compute Mel spectrograms for the reference and background signals

  reference_spectro = compute_mel_spectrogram(reference_signal, sampling_rate)
  background_spectro = compute_mel_spectrogram(background_signal, sampling_rate)

  # Compute binary time-frequency masks defining time-frequency regions
  # where the two signals present significant energy
  reference_mask = compute_tf_mask_significant_energy(reference_spectro,
                                                      energy_threshold=0.9999,
                                                      display_mask=False)

  background_mask = compute_tf_mask_significant_energy(background_spectro,
                                                       energy_threshold=0.9999,
                                                       display_mask=False)

  # Compute time-frequency SNRs on these time-frequency regions
  tf_snr = np.divide(reference_spectro, background_spectro + np.finfo(float).eps)
  if only_mask_with_background:
    tf_snr = np.multiply(background_mask, tf_snr)
  elif only_mask_with_reference:
    tf_snr = np.multiply(reference_mask, tf_snr)
  else:  
    tf_snr = np.multiply(np.multiply(reference_mask, background_mask), tf_snr)


  # Reshape the SNR array and convert it to dB scale
  snr = np.reshape(tf_snr, -1)
  snr = 10.0 * np.log10(snr[snr > 0])

  # Compute gain that produces a median 0dB SNR on these time-frequency regions
  if len(snr) == 0:
    background_gain = 1.
  else:
    background_gain = math.sqrt(10**(np.median(snr - target_snr) / 10.))

  # If display is True, plot the Mel spectrograms and SNR map for debugging
  if display:
    plt.figure()
    plt.subplot(311)
    plt.imshow(10 * np.log10(np.multiply(reference_spectro,
                                         reference_mask)[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(10 * np.log10(np.multiply(background_spectro,
                                         background_mask)[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(10 * np.log10(tf_snr[:, :, 0]),
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()

    plt.show()

  return background_gain


def compute_background_gains_for_unbalanced_mix(reference_signal: np.array,
                                                background_signals: list,
                                                sampling_rate: int,
                                                target_snr: float):
  """
  Compute individual gains for multiple background signals such that
  their weighted sum with the reference signal achieves the target SNR
  using time-frequency energy analysis.

  Parameters
  ----------
  reference_signal : np.array
      The reference signal
  background_signals : list
      list of the background signals
  sampling_rate : int
      sample rate of the signals
  target_snr : float
      The target SNR (in dBs)

  Returns
  -------
  list
      List with the gains to be applied in the background signals

  Raises
  ------
  ValueError
      _description_
  ValueError
      _description_
  """

  if reference_signal.ndim == 1 and len(reference_signal.shape == 1):
    reference_signal = reference_signal.reshape(1, len(reference_signal))

  num_backgrounds = len(background_signals)

  # Check if all background signals have the same shape as the reference signal
  for bg_signal in background_signals:


    if reference_signal.ndim == 1 and len(reference_signal.shape == 1):
      reference_signal = reference_signal.reshape(1, len(reference_signal))

    if reference_signal.shape != bg_signal.shape:
      raise ValueError('All background signals must have the same shape '
                       'as the reference signal. Found shapes:\n'
                       f'Reference: {reference_signal.shape}\n'
                       f'Background: {bg_signal.shape}')

  # Compute Mel spectrograms
  reference_spectro = compute_mel_spectrogram(reference_signal, sampling_rate)
  background_spectros = [compute_mel_spectrogram(bg, sampling_rate)
                         for bg in background_signals]

  # Compute significant energy masks
  reference_mask = compute_tf_mask_significant_energy(
    reference_spectro, energy_threshold=0.999)

  background_masks = [compute_tf_mask_significant_energy(
    bg_spectro, energy_threshold=0.999) for bg_spectro in background_spectros]

  # Compute total background spectrogram by summing all background signals
  combined_background_spectro = sum(background_spectros)
  combined_background_mask = np.clip(sum(background_masks), 0, 1)

  # Compute time-frequency SNR for the combined background
  tf_snr_combined = np.divide(
    reference_spectro, combined_background_spectro + np.finfo(float).eps)

  tf_snr_combined = np.multiply(
    np.multiply(reference_mask, combined_background_mask), tf_snr_combined)

  # Compute the overall gain (same as your original method)
  overall_gain = compute_background_gain_for_unbalanced_mix(
    reference_signal=reference_signal,
    background_signal=np.sum(background_signals, axis=0),
    sampling_rate=sampling_rate,
    target_snr=target_snr,
    display=False
  )

  # Objective function to minimize
  def objective(gains):
    # Calculate the weighted sum of background signals
    weighted_sum = np.sum(np.array(background_signals) *
                          gains[:, np.newaxis], axis=0)

    # Calculate the target output using the overall gain applied to the
    # sum of background signals
    target_output = overall_gain * np.sum(background_signals, axis=0)

    # Return the squared error between the weighted sum and the target output
    return np.sum((weighted_sum - target_output) ** 2)

  # Initial guess for the gains (all set to 1 initially)
  initial_gains = np.ones(num_backgrounds)

  # Minimize the objective function with relaxed tolerance and vectorized computation
  result = minimize(objective, initial_gains, method='L-BFGS-B', options={
    'gtol': 1e-1,  # Gradient tolerance (allowing small error)
    'ftol': 1e-1,  # Function tolerance (allowing small error)
    'maxiter': 100, # Max iterations to speed up
  })

  # If the optimization is successful, return the optimized gains
  if result.success:
    return result.x
  else:
    raise ValueError('Optimization failed to find the background gains.')
