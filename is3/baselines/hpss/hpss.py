""" 
HPSS Baseline
"""
from librosa.decompose import hpss 
from librosa import stft, istft

class HarmonicPercussiveDecomposition:
  def __init__(
    self,
    nfft, 
    window_size,
    overlap,
    margin
  ):
    
    self.nfft = nfft
    self.window_size = window_size
    self.overlap = overlap 
    self.hop_size = hop_size = int(nfft * (1. - overlap))
    self.margin = margin 
    
  def stft_func(self, x):
    spec = stft(
      y=x,
      n_fft=self.nfft, 
      hop_length=self.hop_size,
      win_length=self.nfft, 
      window="hann",
      center=True
    )
    return spec
  
  def istft_func(self, S, length):
    y = istft(
      stft_matrix=S,
      hop_length=self.hop_size,
      win_length=self.nfft, 
      n_fft=self.nfft,
      window="hann",
      center=True,
      length=length
    )
    return y 
  
  def forward(self, x):
    
    S = stft(x)
    mask_h, mask_p = hpss(
      S=S,
      mask=True,
      margin=self.margin
    )
    
    S_p = S*mask_p 
    if self.margin > 1.0:
      S_h = S*(1-mask_p)
    else:
      S_h = S*mask_h
    
    y_h = istft(S_h, length=x.shape[-1])
    y_p = istft(S_p, length=x.shape[-1])
    
    return y_p, y_h, S_p, S_h 