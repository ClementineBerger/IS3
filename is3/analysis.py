"""Utils from spectral analysis : STFT, ISTFT etc..."""

import torch
import torch.nn as nn
import torchaudio.transforms as T

def safe_log(x, eps=1e-12):
  """
  Computes the logarithm base 10 of the input tensor after applying a ReLU activation and adding a small epsilon value for numerical stability.

  Parameters
  ----------
  x : torch.Tensor
    Input tensor for which the logarithm is to be computed.
  eps : float, optional
    A small value added to the input to avoid taking the logarithm of zero. Default is 1e-12.

  Returns
  -------
  torch.Tensor
    The logarithm base 10 of the input tensor after applying ReLU and adding epsilon.
  """
  return torch.log10(nn.ReLU()(x) + eps)

def pad_for_stft(audio, nfft, hop_size):
  """
  Padding function for stft.

  Parameters
  ----------
  audio : torch.Tensor [batch_size, nb_timesteps]
      Audio waveforms in batch
  nfft : _type_
      _description_
  hop_size : _type_
      _description_
  """

  audio_size = audio.shape[-1]
  num_frames = audio_size // hop_size

  pad_size = max(0, nfft + (num_frames - 1) * hop_size - audio_size)

  if pad_size == 0:
    return audio

  else:
    audio = torch.nn.functional.pad(audio, pad=(0, pad_size))
    return audio


def stft(
        audio: torch.Tensor,
        nfft: int,
        overlap: float,
        window_size=None,
        center=True,
        pad_end=True):
  """
  Differentiable stft in pytorch, computed in batch.

  Parameters
  ----------
  audio : torch.Tensor
  nfft : int
      Size of Fourier transform.
  overlap : float
      Portion of overlapping window
  center : bool, optional
      by default False
  pad_end : bool, optional
      Padding applied to the audio or not, by default True

  Returns
  -------
  torch.Tensor
      stft [batch_size, nfft//2 + 1, n_frames]
  """

  hop_size = int(nfft * (1. - overlap))

  if pad_end:
    audio = pad_for_stft(audio=audio, nfft=nfft, hop_size=hop_size)
  # pb du center et de la istft à régler, est-ce que ça change qqch pour le
  # padding ?

  if window_size is None:
    window_size = nfft

  window = torch.hann_window(window_size).to(device=audio.device)

  spectrogram = torch.stft(
      input=audio,
      n_fft=nfft,
      hop_length=hop_size,
      win_length=window_size,
      window=window,
      center=center,
      return_complex=True
  ) 

  return spectrogram.transpose(-2, -1)


def istft(
        stft: torch.Tensor,
        nfft: int = 2048,
        window_size=None,
        overlap=0.75,
        center=True,
        length=None):
  """Differentiable istft in PyTorch, computed in batch."""

  # input stft [batch_size, n_frames, nfft//2 + 1], need to transpose the
  # time and frequency dimensions

  stft = stft.transpose(-2, -1)
  hop_length = int(nfft * (1.0 - overlap))

  initial_shape = stft.shape
  
  if window_size is None:
    window_size = nfft

  if len(initial_shape) > 3:
    stft = stft.reshape(-1, initial_shape[-2], initial_shape[-1])

  assert nfft * overlap % 2.0 == 0.0
  window = torch.hann_window(int(nfft), device=stft.device)
  s = torch.istft(
      input=stft,
      n_fft=int(nfft),
      hop_length=hop_length,
      win_length=window_size,
      window=window,
      center=center,
      length=length,
      onesided=True,
      return_complex=False)

  if len(initial_shape) > 3:
    s = s.reshape(initial_shape[:-2] + (s.shape[-1],))

  return s