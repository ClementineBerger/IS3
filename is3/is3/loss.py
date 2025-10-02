"""
Loss functions.
"""

from typing import Tuple

import torch 
import torch.nn as nn

from torch.autograd import Function
import torch.nn.functional as F

from model import STFTFB

class angle(Function):
  """Similar to torch.angle but robustify the gradient for zero magnitude."""

  @staticmethod
  def forward(ctx, x: torch.Tensor):
    ctx.save_for_backward(x)
    return torch.atan2(x.imag, x.real)

  @staticmethod
  def backward(ctx, grad: torch.Tensor):
    (x,) = ctx.saved_tensors
    grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
    return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))  

class SpectralLoss(nn.Module):
  def __init__(self, 
               gamma: float=0.3, 
               factor_magnitude: float=1000, 
               factor_complex: float=1000, 
               factor_under: float=1,
               weight_side_chain=None,
               normalize=False):
    super().__init__()
    self.gamma = gamma # compression factor
    self.factor_magnitude = factor_magnitude  #lambda_magnitude
    self.factor_complex = factor_complex  #lambda_complex
    self.factor_under = factor_under
    self.weight_side_chain = weight_side_chain
    self.normalize = normalize
    
  def forward(self, input, target, side_chain_signal=None):
    # input: [B, C, T, F]
    # target: [B, C, T, F]

    # input = torch.view_as_complex(input)
    # target = torch.view_as_complex(target)
    # side chain signal = SPECTROGRAM of the side chain signal [B, C, T, F]
    
    input_abs = torch.abs(input)
    target_abs = torch.abs(target)
    
    if self.normalize:
      norm_coeffs = torch.mean(input_abs.pow(2), dim=(1, 2, 3), keepdim=True)
    else: 
      norm_coeffs = 1.
    
    if self.weight_side_chain is not None and side_chain_signal is not None:
      side_chain_signal_abs = torch.abs(side_chain_signal)  # [B, 1, T, F]
      max_side_chain = torch.amax(side_chain_signal_abs, dim=(1, 2, 3), keepdim=True)
      weights_values = 1. + self.weight_side_chain * side_chain_signal_abs / max_side_chain
      with torch.no_grad():
        weights = torch.where(side_chain_signal_abs > 0.001*max_side_chain, weights_values, 1.)
    else: 
      with torch.no_grad():
        weights = torch.ones_like(input_abs).to(input.device)
    
    if self.gamma != 1.:
      # Apply compression
      input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
      target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
    
    
    magnitude_loss = torch.mean(weights*(input_abs - target_abs).pow(2)/norm_coeffs)

    
    if self.factor_under != 1.:
      magnitude_loss *= torch.where(input_abs < target_abs, self.factor_under, 1.)
      
    magnitude_loss = torch.mean(magnitude_loss) * self.factor_magnitude

    if self.factor_complex != 1.:
      input = weights * input_abs * torch.exp(1j * angle.apply(input))
      target = weights * target_abs * torch.exp(1j * angle.apply(target))
      

    complex_loss = F.mse_loss(
      torch.view_as_real(input)/norm_coeffs, target=torch.view_as_real(target)/norm_coeffs 
    )* self.factor_complex      
    
    spec_loss = magnitude_loss + complex_loss

    return spec_loss, (magnitude_loss, complex_loss)
  
class MultiResSpecLoss(nn.Module):
  def __init__(
      self,
      n_ffts: Tuple[int] = (256, 512, 1024),
      gamma: float = 0.3,
      factor_magnitude: float = 500,
      factor_complex: float = 500,
      weight_side_chain=None,
      normalize=False,
      # normalized=True, pour l'instant on ne l'utilise pas, à voir
  ):    
    
    super().__init__()
    
    self.n_ffts = n_ffts
    self.gamma = gamma
    self.factor_magnitude = factor_magnitude  
    
    self.stfts = [STFTFB(nfft=n_fft, overlap=0.75, window_size=n_fft, window=None, center=True) for n_fft in n_ffts]
    
    self.factor_complex = [factor_complex] * len(self.stfts)
    
    self.weight_side_chain = weight_side_chain
    
    self.normalize = normalize
    
  def forward(self, input, target, side_chain_signal=None):
    
    # input: [B, 1, T] -> waveforms
    # target: [B, 1, T]
    # side_chain_signal: [B, 1, T] -> waveform also
    
    loss = torch.zeros((), dtype=input.dtype, device=input.device)
    
    for i, stft in enumerate(self.stfts):
      input_spec = stft(input.squeeze(1)) # remove channel dimension
      target_spec = stft(target.squeeze(1)) # remove channel dimension
      input_spec_abs = input_spec.abs()
      target_spec_abs = target_spec.abs()  
      
      if self.normalize:
          norm_coeffs = torch.mean(input_spec_abs.pow(2), dim=(1, 2, 3), keepdim=True)
      else: 
        norm_coeffs = 1.
      
      if self.weight_side_chain is not None and side_chain_signal is not None:
        side_chain_signal_spec = stft(side_chain_signal.squeeze(1))
        side_chain_signal_abs = torch.abs(side_chain_signal_spec)  # [B, 1, T, F]
        max_side_chain = torch.amax(side_chain_signal_abs, dim=(1, 2, 3), keepdim=True)
        weights_values = 1. + self.weight_side_chain * side_chain_signal_abs / max_side_chain
        with torch.no_grad():
          weights = torch.where(side_chain_signal_abs > 0.001*max_side_chain, weights_values, 1.)
      else: 
        with torch.no_grad():
          weights = torch.ones_like(input_spec_abs).to(input.device)        
      
      if self.gamma != 1.:
        input_spec_abs = weights * input_spec_abs.clamp_min(1e-12).pow(self.gamma)
        target_spec_abs = weights * target_spec_abs.clamp_min(1e-12).pow(self.gamma)
      
      loss += F.mse_loss(input_spec_abs/norm_coeffs, target_spec_abs/norm_coeffs) * self.factor_magnitude
      
      input_spec = input_spec_abs * torch.exp(1j * angle.apply(input_spec))/norm_coeffs
      target_spec = target_spec_abs * torch.exp(1j * angle.apply(target_spec))/norm_coeffs
      
      loss += F.mse_loss(torch.view_as_real(input_spec), torch.view_as_real(target_spec)) * self.factor_complex[i]
      
    return loss
  
  
class Loss(nn.Module):
  def __init__(
    self,
    gamma_spec: float = 0.3,
    f_mag_spec: float = 1000,
    f_complex_spec: float = 1000,
    f_under_spec: float = 1,
    n_ffts: Tuple[int] = (256, 512, 1024),
    gamma_mr: float = 0.3,
    f_mag_mr: float = 500,
    f_complex_mr: float = 500,
    stft_params=None,  # for istft
    weight_impulse: float = 1,
    weight_stationary: float = 1,
    weight_mixture: float = 1,
    weight_stationary_when_impulse = None,
    normalize=False,
  ):
    
    super().__init__()
    
    self.mr_loss = MultiResSpecLoss(
      n_ffts,
      gamma_mr,
      f_mag_mr,
      f_complex_mr,
      weight_side_chain=weight_stationary_when_impulse,
      normalize=normalize
    )
    
    self.spec_loss = SpectralLoss(
      gamma_spec,
      f_mag_spec,
      f_complex_spec,
      f_under_spec,
      weight_side_chain=weight_stationary_when_impulse,
      normalize=normalize 
    )
    
    self.stft = STFTFB(**stft_params)    
    
    self.weight_impulse = weight_impulse
    self.weight_stationary = weight_stationary
    self.weight_mixture = weight_mixture
    self.weight_stationary_when_impulse = weight_stationary_when_impulse
    
    self.normalize = normalize
    
    
  def forward_composant(self, estimate_spec, target_spec, side_chain_signal=None, normalize=False):
    """
    Parameters
    ----------
    estimate_spec : torch.Tensor
      Estimated complex spectrogram of the enhanced speech.
      Shape: [B, 1, F, T].
    target_spec : torch.Tensor
      Target complex spectrogram of the clean speech.
      Shape: [B, 1, F, T].
    side_chain_signal : torch.Tensor, optional
      Side chain signal complex spectrogram.
    """
    
    if self.weight_stationary_when_impulse is not None and side_chain_signal is not None:
      side_chain_waveform = self.stft.inverse(side_chain_signal)
    else: 
      side_chain_waveform = None
    
    loss_spec, (loss_mag, loss_complex) = self.spec_loss(estimate_spec, target_spec, side_chain_signal)
    
    estimate_waveform = self.stft.inverse(estimate_spec)
    target_waveform = self.stft.inverse(target_spec)
    loss_mr = self.mr_loss.forward(
      input=estimate_waveform, 
      target=target_waveform,
      side_chain_signal=side_chain_waveform,
    )
    
    loss = loss_spec + loss_mr
    
#    return loss, (loss_spec, loss_mr, loss_mag, loss_complex)
    return loss, (loss_spec, loss_mr)
  
  def combine_losses(self, loss_impulse, loss_stationary, loss_mixture):
    return self.weight_impulse * loss_impulse + self.weight_stationary * loss_stationary + self.weight_mixture * loss_mixture
  
  def forward(
    self, 
    estimate_imp_spec,
    estimate_sta_spec,
    target_imp_spec,
    target_sta_spec,
    target_mix_spec
  ):
    
    loss_imp, (loss_imp_spec, loss_imp_mr) = self.forward_composant(estimate_imp_spec, target_imp_spec, side_chain_signal=None, normalize=self.normalize)
    loss_sta, (loss_sta_spec, loss_sta_mr) = self.forward_composant(estimate_sta_spec, target_sta_spec, side_chain_signal=target_imp_spec, normalize=self.normalize)
    loss_mix, (loss_mix_spec, loss_mix_mr) = self.forward_composant(estimate_imp_spec + estimate_sta_spec, target_mix_spec, side_chain_signal=None, normalize=self.normalize)
    
    loss = self.combine_losses(loss_imp, loss_sta, loss_mix)
    
    loss_dict = {
      "imp":{
        "overall": loss_imp,
        "spec": loss_imp_spec,
        "mr": loss_imp_mr
      },
      "sta":{
        "overall": loss_sta,
        "spec": loss_sta_spec,
        "mr": loss_sta_mr
      },
      "mix":{
        "overall": loss_mix,
        "spec": loss_mix_spec,
        "mr": loss_mix_mr
      },
    }
    
    return loss, loss_dict
    
    
if __name__ == "__main__":
  
  # Test
  
  stft_params = {
    "nfft": 2048,
    "overlap": 0.75,
    "window_size": 2048,
    "window": None,
    "center": True
  }  
  
  loss = Loss(stft_params=stft_params)
  

  estimate_spec = torch.view_as_complex(torch.randn(2, 1, 1025, 257, 2))
  target_spec = torch.view_as_complex(torch.randn(2, 1, 1025, 257, 2))
  
  loss_value, (loss_spec, loss_mr, loss_mag, loss_complex) = loss.forward_composant(estimate_spec, target_spec)
  
  print(loss_value)
  print(loss_spec)
  print(loss_mr)
  print(loss_mag)
  print(loss_complex)
  