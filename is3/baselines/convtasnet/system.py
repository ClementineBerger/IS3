import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn

from torch import Tensor

from is3.analysis import pad_for_stft, stft, istft



class STFTFB(nn.Module):
  """STFT filterbanks using torch.stft.

  Parameters
  ----------
  n_fft : int
    The number of FFT filters.
  kernel_size : int
    The size of the FFT filters.
  stride : int
    The hop size of the STFT.
  window Tensor, default=None
    the window for overlap add to use. None means 
    it is treated as if having 1 everywhere in the window.

  Methods
  -------
  inverse(spec)
    From Time Frequency to time domain.
  forward(x)
    From time domain to Time Frequency.

  Examples
  --------
  >>> import torch
  >>> from speech_enhancement.dsp.stft import STFTFB
  >>> x = torch.randn(2, 16000)
  >>> fb = STFTFB(512, 512, 256)
  >>> spec = fb(x)
  >>> spec.shape

  """

  def __init__(
      self,
      nfft,
      overlap,
      window_size,
      window=None,
      center=True,

  ):

    super().__init__()
    self.nfft = nfft
    self.overlap = overlap
    self.window_size = window_size
    self.window = window
    self.center = center
    
    
  def forward(self, x: Tensor) -> Tensor:
    """Computes STFT

    Parameters
    ----------
    x : Tensor
      Time domain tensor of shape [B,T]

    Returns
    -------
    Tensor
      the spectrogram of the input as a complex tensor 
      with shape [B,1,F,T]

    """

    spec = stft(
      audio=x, nfft=self.nfft, overlap=self.overlap, window_size=self.window_size, center=self.center
    ).transpose(-1, -2) # [B,F,T]
    
    return spec.unsqueeze(1)  # complex [B,1,F,T]

  def inverse(self, spec: Tensor, length=None) -> Tensor:
    """
    Computes ISTFT

    Parameters
    ----------
    spec : Tensor of shape [B,C,F,T]
      The spectrogram as a complex tensor
    length : int, default=None
      The length of the output signal.

    Returns
    -------
    Tensor
      The time domain tensor of shape [B,T]
    """
    
    x = istft(
      stft=spec.squeeze(1).transpose(-1, -2), nfft=self.nfft, overlap=self.overlap, window_size=self.window_size, center=self.center, length=length
    )  
    
    return x


class System(pl.LightningModule):
  """
  Pytorch lightning system to train the model and log useful informations.
  """

  default_monitor: str = "val/loss"

  def __init__(self,
               model,
               optimizer,
               scheduler,
               loss_func,
               train_loader,
               val_loader,
               config) -> None:
    super().__init__()

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.conf = config
    
    self.stft = STFTFB(**self.conf['stft_params'])

    self.loss_func = loss_func
    
  def forward(self, x):
    y_i_spec, y_s_spec = self.model(x)
    return y_i_spec, y_s_spec
  
  def configure_optimizers(self):
    """Initialize optimizers"""

    if self.scheduler is None:
      return self.optimizer

    else:
      return [self.optimizer], [{"scheduler": self.scheduler,
                                 "interval": "step"}]  
      
  def training_step(self, batch, batch_idx):
    """
    Training step
    """
    
    # Target
    x, y_i, y_s = batch

    targets = torch.stack([y_i, y_s], dim=-2)  # [B,2,T]
    
    # Prediction
    prediction = self.model.forward(x)
    
    loss = self.loss_func(prediction, targets)

    self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    # for name, param in self.named_parameters():
    #     if param.grad is None:
    #         print(f"Parameter not used in training: {name}")    
    
    return loss      
      
  # def training_step(self, batch, batch_idx):
  #   """
  #   Training step
  #   """
    
  #   # Target
  #   x, y_i, y_s = batch
  #   x_spec_target = self.stft(x)
  #   y_i_spec_target = self.stft(y_i)
  #   y_s_spec_target = self.stft(y_s)
    
  #   # Prediction
  #   prediction = self.model.forward(x)
  #   y_hat_i = prediction[..., 0, :]
  #   y_hat_s = prediction[..., 1, :]
  #   y_i_spec = self.stft(y_hat_i)
  #   y_s_spec = self.stft(y_hat_s) 

  #   loss, loss_dict = self.loss_func(
  #     estimate_imp_spec=y_i_spec,
  #     estimate_sta_spec=y_s_spec,
  #     target_imp_spec=y_i_spec_target,
  #     target_sta_spec=y_s_spec_target,
  #     target_mix_spec=x_spec_target,
  #   )

  #   #logging
  #   loss_imp = loss_dict['imp']['overall']
  #   loss_imp_spec = loss_dict['imp']['spec']
  #   loss_imp_mr = loss_dict['imp']['mr']
  #   loss_sta = loss_dict['sta']['overall']
  #   loss_sta_spec = loss_dict['sta']['spec']
  #   loss_sta_mr = loss_dict['sta']['mr']
  #   loss_mix = loss_dict['mix']['overall']
  #   loss_mix_spec = loss_dict['mix']['spec']
  #   loss_mix_mr = loss_dict['mix']['mr']
    
  #   self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("train/impulse/overall", loss_imp, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/impulse/spec", loss_imp_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/impulse/mr", loss_imp_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("train/stationary/overall", loss_sta, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/stationary/spec", loss_sta_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/stationary/mr", loss_sta_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("train/mix/overall", loss_mix, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/mix/spec", loss_mix_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("train/mix/mr", loss_mix_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   return loss
  
  def validation_step(self, batch, batch_idx):
    """
    Training step
    """
    
    # Target
    x, y_i, y_s = batch

    targets = torch.stack([y_i, y_s], dim=-2)  # [B,2,T]
    
    # Prediction
    prediction = self.model.forward(x)
    
    loss = self.loss_func(prediction, targets)

    self.log("val/loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    return loss    
  
  # def validation_step(self, batch, batch_idx):
    
  #   # Target
  #   x, y_i, y_s = batch
  #   x_spec_target = self.stft(x)
  #   y_i_spec_target = self.stft(y_i)
  #   y_s_spec_target = self.stft(y_s)
    
  #   # Prediction
  #   prediction = self.model.forward(x)
  #   y_hat_i = prediction[..., 0, :]
  #   y_hat_s = prediction[..., 1, :]
  #   y_i_spec = self.stft(y_hat_i)
  #   y_s_spec = self.stft(y_hat_s)
    
  #   loss, loss_dict = self.loss_func(
  #     estimate_imp_spec=y_i_spec,
  #     estimate_sta_spec=y_s_spec,
  #     target_imp_spec=y_i_spec_target,
  #     target_sta_spec=y_s_spec_target,
  #     target_mix_spec=x_spec_target,
  #   )

  #   #logging
  #   loss_imp = loss_dict['imp']['overall']
  #   loss_imp_spec = loss_dict['imp']['spec']
  #   loss_imp_mr = loss_dict['imp']['mr']
  #   loss_sta = loss_dict['sta']['overall']
  #   loss_sta_spec = loss_dict['sta']['spec']
  #   loss_sta_mr = loss_dict['sta']['mr']
  #   loss_mix = loss_dict['mix']['overall']
  #   loss_mix_spec = loss_dict['mix']['spec']
  #   loss_mix_mr = loss_dict['mix']['mr']
    
  #   self.log("val/loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("val/impulse/overall", loss_imp, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/impulse/spec", loss_imp_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/impulse/mr", loss_imp_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("val/stationary/overall", loss_sta, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/stationary/spec", loss_sta_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/stationary/mr", loss_sta_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   self.log("val/mix/overall", loss_mix, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/mix/spec", loss_mix_spec, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
  #   self.log("val/mix/mr", loss_mix_mr, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
  #   return loss
  
  def on_validation_epoch_end(self) -> None:
    
    tensorboard = self.logger.experiment
    
    batch = next(iter(self.val_dataloader()))
    
    x, y_i, y_s = batch
    x = x.to('cuda')
    y_i = y_i.to('cuda')
    y_s = y_s.to('cuda')
    
    # Prediction
    prediction = self.model.forward(x)
    y_i_predicted = prediction[..., 0, :]
    y_s_predicted = prediction[..., 1, :]
    
    # choosing 3 samples
    idx = [0, x.shape[0]//2, x.shape[0]-1]
    
    # Plotting the waveforms
    fig, ax = plt.subplots(len(idx), 3, figsize=(15, 10))
    plt.suptitle("Waveforms")
    
    t = np.arange(x.shape[-1])/self.conf['sr']
    
    for i in range(len(idx)):
      
      ax[i, 0].plot(t, y_i[idx[i], :].detach().cpu().numpy(), color='skyblue', label='ground_truth')
      ax[i, 0].plot(t, y_i_predicted[idx[i], :].detach().cpu().numpy(), alpha=0.6, color='orange', label='predicted')
      ax[i, 0].set_title("Impulse")
      
      ax[i, 1].plot(t, y_s[idx[i], :].detach().cpu().numpy(), color='skyblue', label='ground_truth')
      ax[i, 1].plot(t, y_s_predicted[idx[i], :].detach().cpu().numpy(), alpha=0.6, color='orange', label='predicted')
      ax[i, 1].set_title("Background")
      
      ax[i, 2].plot(t, x[idx[i], :].detach().cpu().numpy(), color='skyblue', label='ground_truth')
      ax[i, 2].plot(t, y_i_predicted[idx[i], :].detach().cpu().numpy() + y_s_predicted[idx[i], :].detach().cpu().numpy(), alpha=0.6, color='orange', label='predicted')
      ax[i, 2].set_title("Mixture")
      
    tensorboard.add_figure("waveforms", fig, global_step=self.current_epoch)
    
    # Playing the audios
    
    for i in range(len(idx)):
      norm = torch.maximum(
        torch.max(torch.abs(x[idx[i]])),
        torch.max(torch.abs(y_i_predicted[idx[i]] + y_s_predicted[idx[i]]))
      )
      
      tensorboard.add_audio(f"predicted/impulse/{i}", y_i_predicted[idx[i]]/norm, self.current_epoch, self.conf['sr'])
      tensorboard.add_audio(f"predicted/background/{i}", y_s_predicted[idx[i]]/norm, self.current_epoch, self.conf['sr'])
      tensorboard.add_audio(f"predicted/mixture/{i}", (y_i_predicted[idx[i]] + y_s_predicted[idx[i]])/norm, self.current_epoch, self.conf['sr'])
      
      tensorboard.add_audio(f"target/impulse/{i}", y_i[idx[i]]/norm, self.current_epoch, self.conf['sr'])
      tensorboard.add_audio(f"target/background/{i}", y_s[idx[i]]/norm, self.current_epoch, self.conf['sr'])
      tensorboard.add_audio(f"target/mixture/{i}", x[idx[i]]/norm, self.current_epoch, self.conf['sr'])
      
    return super().on_validation_epoch_end()
  
  def train_dataloader(self):
    """Training dataloader"""
    return self.train_loader

  def val_dataloader(self):
    """Validation dataloader"""
    return self.val_loader

  def on_save_checkpoint(self, checkpoint):
    """Overwrite if you want to save more things in the checkpoint."""
    checkpoint["training_config"] = self.conf
    return checkpoint