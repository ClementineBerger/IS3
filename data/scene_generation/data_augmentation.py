import torch
import torchaudio
import numpy as np
import random

from torchaudio.functional import equalizer_biquad

class AudioAugmentor:
  """Class for data augmentation
  """
  def __init__(self, sample_rate: int = 16000, augment_scale: int = 5):
    """
    Initialize the AudioAugmentor with common parameters.

    Parameters
    ----------
    sample_rate : int, optional
      The sample rate of the audio, by default 16000
    augment_scale : int, optional
      A value between 1 and 5 to control the power of the augmentation,
      by default 5
    """
    self.sample_rate = sample_rate
    self.augment_scale = augment_scale

  def time_shift(self,
                 utt: np.ndarray,
                 augment_scale: int = None) -> np.ndarray:
    """
    Apply a time shift to the input audio array to augment the data.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment.
    augment_scale : int, optional
      Override the class-level augment scale.

    Returns
    -------
    np.ndarray
      The augmented audio array.
    """
    if augment_scale is None:
      augment_scale = self.augment_scale

    utt_tensor = torch.from_numpy(utt)
    with torch.no_grad():
      scale = random.randint(1, augment_scale)
      shift_scale = scale / 1000
      utt_split = int(utt_tensor.shape[1] * shift_scale)
      utt_aug = torch.cat([utt_tensor[:, -utt_split:],
                           utt_tensor[:, :-utt_split]],
                           dim=-1)

    return utt_aug.numpy()

  def eq_aug(self,
             utt: np.ndarray,
             max_n_bands: int = 5,
             min_n_bands: int = 1,
             fmax_band: int = 8000,
             fmin_band: int = 50,
             amp_db: int = 5,
             augment_scale: int = None) -> np.ndarray:
    """
    Apply random equalization (EQ) to the input audio.

    This function introduces random equalization by applying a series of
    EQ bands to the audio signal, each with a randomly chosen center
    frequency and gain. The number of EQ bands, their frequencies, and
    the amplitude adjustments are all randomly determined within
    specified ranges.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment.
    max_n_bands : int, optional
      Maximum number of EQ bands, by default 5.
    min_n_bands : int, optional
      Minimum number of EQ bands, by default 1.
    fmax_band : int, optional
      Maximum frequency of the EQ, by default 8000.
    fmin_band : int, optional
      Minimum frequency of the EQ, by default 50.
    amp_db : int, optional
      Maximum amplitude change in dB, by default 5.
    augment_scale : int, optional
      Override the class-level augment scale.

    Returns
    -------
    np.ndarray
      The augmented audio array.
    """
    if augment_scale is None:
      augment_scale = self.augment_scale

    assert max_n_bands > min_n_bands

    utt_tensor = torch.from_numpy(utt)
    n_bands = torch.randint(low=min_n_bands, high=max_n_bands, size=(1,))

    center_freqs = torch.randint(low=fmin_band, high=fmax_band, size=(n_bands,))
    gains = (torch.rand(n_bands) * (2 * amp_db) - amp_db) * augment_scale / 5

    for i in range(n_bands):
      utt_tensor = equalizer_biquad(waveform=utt_tensor,
                                    sample_rate=self.sample_rate,
                                    center_freq=center_freqs[i],
                                    gain=gains[i],
                                    Q=0.7)

    return utt_tensor.numpy()

  def compression(self,
                  utt: np.ndarray,
                  augment_scale: int = None) -> np.ndarray:
    """
    Compress the input audio to reduce dynamic range.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment.
    augment_scale : int, optional
      Override the class-level augment scale.

    Returns
    -------
    np.ndarray
      The compressed audio array.
    """
    if augment_scale is None:
      augment_scale = self.augment_scale

    utt_tensor = torch.from_numpy(utt)
    with torch.no_grad():
      scale = augment_scale
      enhancement_amount = np.random.randint(low=0, high=scale * 2)
      utt_tensor = torchaudio.functional.contrast(utt_tensor,
                                                  enhancement_amount)

    return utt_tensor.numpy()

  #no noise
  def noise(self,
            utt: np.ndarray,
            sample_rate: int = None,
            augment_scale: int = None) -> np.ndarray:
    """
    Add random noise to the input audio to augment the data.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment. Expected to have shape (n_channels, n_samples).
    sample_rate : int, optional
      Sample rate of the audio. If not provided, the class-level
      `sample_rate` is used.
    augment_scale : int, optional
      A value between 1 and 5 to control the intensity of the noise
      augmentation. If not provided, the class-level `augment_scale` is used.

    Returns
    -------
    np.ndarray
      The augmented audio array.
    """
    if sample_rate is None:
      sample_rate = self.sample_rate
    if augment_scale is None:
      augment_scale = self.augment_scale

    utt_tensor = torch.from_numpy(utt)
    scale = augment_scale
    sr = sample_rate
    snr_min = 12
    snr_max = 100 * 1 / scale

    with torch.no_grad():
      noise = torch.normal(torch.zeros(sr),
                           torch.ones(sr)).to(utt_tensor.device)

      f_decay = np.random.choice([0, 1, 2, -1, -2])
      spec = torch.fft.rfft(noise)
      mask = torch.pow(torch.linspace(1, (sr / 2) ** 0.5,
                       spec.shape[0]).to(utt_tensor.device), -f_decay)

      spec *= mask
      noise = torch.fft.irfft(spec).reshape(1, -1).squeeze()
      noise /= torch.sqrt(torch.mean(torch.square(noise)))

      noise = torch.cat([noise] * int(
        np.ceil(utt_tensor.shape[1] / noise.shape[0])),
        axis=0)[:utt_tensor.shape[1]]

      snr = np.random.uniform(snr_min, snr_max)
      gain = torch.sqrt(torch.square(utt_tensor).sum() /
                        (10 ** (snr / 10) * torch.square(noise).sum()))
      noise *= gain

      utt_tensor += noise

    return utt_tensor.numpy()

  def overdrive(self,
                utt: np.ndarray,
                min_gain: float = 0.3,
                max_gain: float = 1.0,
                min_colour: float = 0.3,
                max_colour: float = 1.0,
                augment_scale: int = None) -> np.ndarray:
    """
    Apply overdrive effect to the input audio.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment. Expected to have shape (n_channels, n_samples).
    max_n_bands : int, optional
      Maximum number of EQ bands to apply, by default 5.
    min_n_bands : int, optional
      Minimum number of EQ bands to apply, by default 1.
    fmax_band : int, optional
      Maximum center frequency for the EQ bands, by default 8000 Hz.
    fmin_band : int, optional
      Minimum center frequency for the EQ bands, by default 50 Hz.
    amp_db : int, optional
      Maximum amplitude adjustment in dB for each EQ band, by default 5 dB.
    augment_scale : int, optional
      A value between 1 and 5 to control the overall strength of the EQ
      augmentation. If not provided, the class-level `augment_scale` is used.

    Returns
    -------
    np.ndarray
      The augmented audio array with random equalization applied.
    """
    if augment_scale is None:
      augment_scale = self.augment_scale

    utt_tensor = torch.from_numpy(utt)
    with torch.no_grad():
      scale = augment_scale

      gain = np.random.uniform(
        low=min_gain,
        high=min_gain + (max_gain - min_gain) / 5 * scale)

      colour = np.random.uniform(
        low=min_colour,
        high=min_colour + (max_colour - min_colour) / 5 * scale)

      utt_tensor = torchaudio.functional.overdrive(
        utt_tensor, gain=gain, colour=colour)

    return utt_tensor.numpy()


  def generate_params(self, param_config: dict) -> dict:
    """
    Generate parameters for effects based on normal distribution.

    Parameters
    ----------
    param_config : dict
      A dictionary where keys are parameter names and values are tuples
      containing the mean and standard deviation for normal distribution.

    Returns
    -------
    dict
      A dictionary with parameter names and their generated values.
    """
    params = {}
    for param_name, (mean, std) in param_config.items():
      value = np.random.normal(loc=mean, scale=std)
      params[param_name] = value
    return params

  def process_chain(self,
                    utt: np.ndarray,
                    effects_chain: list) -> (np.ndarray, dict):
    """
    Process the input audio through a chain of effects and return the
    augmented audio along with the parameters used for each effect.

    Parameters
    ----------
    utt : np.ndarray
      Audio array to augment.
    effects_chain : list
      A list of tuples, where each tuple contains the name of the effect
      as a string and a dictionary of parameter configurations for that
      effect.

    Returns
    -------
    np.ndarray
      The augmented audio array.
    dict
      A dictionary containing the parameters used for each effect in
      the chain.
    """
    augmented_audio = utt
    params_used = {}

    for effect_name, param_config in effects_chain:
      # Generate parameters based on normal distribution
      effect_params = self.generate_params(param_config)

      # Float to int conv (necessary for "eq_aug" params)
      if effect_name == 'eq_aug':
        effect_params['max_n_bands'] = int(np.round(effect_params['max_n_bands']))
        effect_params['amp_db'] = int(np.round(effect_params['amp_db']))

        # Assure max_n_bands > 1
        effect_params['max_n_bands'] = max(2, effect_params['max_n_bands'])

      # Assure augment scale positive
      if 'augment_scale' in effect_params:
        effect_params['augment_scale'] = max(1, effect_params['augment_scale'])

      # Get the effect method from the class using its name
      effect_method = getattr(self, effect_name)
      augmented_audio = effect_method(augmented_audio, **effect_params)
      params_used[effect_name] = effect_params

    return augmented_audio, params_used
