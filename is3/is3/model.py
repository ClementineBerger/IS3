"""
Deep Filtering model inspired from DeepFilterNet from impulse sound rejection.
"""

import torch
from torch.nn.parameter import Parameter
from torch import Tensor, nn

from functools import partial
from typing import Tuple, Union, Iterable, Callable, Optional
import math
import os

from rendering.dpnmm.analysis import pad_for_stft, stft, istft

class ApplyDF(nn.Module):
  """ Apply deepfiltering to a spectrogram."""

  def __init__(self, order, nb_df, lookahead):
    super().__init__()
    self.df_order = order
    self.lookahead = lookahead
    self.nb_df = nb_df
    self.pad_df = nn.ConstantPad2d(
        (0, 0, order - 1 - lookahead, lookahead), 0.0)

  def forward(self, spec, coefs):
    """Apply deep filtering coefficients to a complex-valued spectrogram 
    tensor.

    Parameters
    ----------
    spec : torch.Tensor, shape [B, 1, T, F, 2]
      The input complex-valued spectrogram tensor.
    coefs : torch.Tensor, shape [B, order, T, F, 2]
      The MF coefficients tensor for each channel and time lag.

    Returns
    -------
    torch.Tensor, shape [B, F, T, 2]
      The output complex-valued spectrogram tensor with the MF 
      coefficients applied.

    """
    spec_u = self.pad_df(torch.view_as_complex(spec))
    spec_u = spec_u.unfold(2, self.df_order, 1)  # kernel size, dilation, padding
    coefs = torch.view_as_complex(coefs)
    spec_f = spec_u.narrow(-2, 0, self.nb_df)
    coefs = coefs.view(coefs.shape[0], -1, self.df_order, *coefs.shape[2:])
    spec_f = torch.einsum("...tfn,...ntf->...tf", spec_f, coefs)
    spec = spec.clone()
    spec[..., : self.nb_df, :] = torch.view_as_real(spec_f)
    return spec

class Conv2dNormAct(nn.Sequential):
  """A combination of Conv2d, normalization, and activation layers in sequence.

  Parameters
  ----------
  in_ch : int
    Number of input channels.
  out_ch : int
    Number of output channels.
  kernel_size : Union[int, Iterable[int]]
    Size of the convolution kernel.
  fstride : int, default=1
    Stride of the convolution on the feature dimension.
  dilation : int, default=1
    Dilation factor for convolution.
  fpad : bool, default=True
    Whether to apply padding on the feature dimension.
  bias : bool, default=True
    Whether to include bias in the convolution.
  separable : bool, default=False
    Whether to use separable convolutions.
  norm_layer : default=[Callable[..., torch.nn.Module]]
    Normalization layer.
  activation_layer : default=[Callable[..., torch.nn.Module]]
    Activation layer.
  causal : bool, default=True.
    Whether to apply causal padding on the time axis.

  Methods
  -------
  forward(x)
    Applies the convolutional layer to the input tensor.

  """

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      kernel_size: Union[int, Iterable[int]],
      fstride: int = 1,
      dilation: int = 1,
      fpad: bool = True,
      bias: bool = True,
      separable: bool = False,
      norm_layer: Optional[Callable[..., torch.nn.Module]
                           ] = torch.nn.BatchNorm2d,
      activation_layer: Optional[Callable[..., torch.nn.Module]
                                 ] = torch.nn.ReLU,
      causal=True,
  ):
    layers = []
    lookahead = 0  # This needs to be handled on the input feature side
    # Padding on time axis
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else tuple(kernel_size)
    )
    if fpad:
      fpad_ = kernel_size[1] // 2 + dilation - 1
    else:
      fpad_ = 0
    if causal:
      pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
    else:
      pad = (0,)
      layers.append(nn.Identity())  # trick to have consistent #lyayers
    if any(x > 0 for x in pad):
      layers.append(nn.ConstantPad2d(pad, 0.0))
    groups = math.gcd(in_ch, out_ch) if separable else 1
    if groups == 1:
      separable = False
    if max(kernel_size) == 1:
      separable = False
    layers.append(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(0, fpad_),
            stride=(1, fstride),  # Stride over time is always 1
            dilation=(1, dilation),  # Same for dilation
            groups=groups,
            bias=bias,
        )
    )
    if separable:
      layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
    if norm_layer is not None:
      layers.append(norm_layer(out_ch))
    if activation_layer is not None:
      layers.append(activation_layer())
    super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
  """A PyTorch sequential block containing a transposed convolutional
  layer with default normalization and activation.

  Parameters
  ----------
  in_ch : int
    Number of input channels.
  out_ch : int
    Number of output channels.
  kernel_size : int or Tuple[int, int]
    Size of the convolutional kernel.
  fstride : int, default=1
    Stride in the frequency (width) dimension of the input.
  dilation : int, default=1
    Dilation rate for the kernel.
  fpad : bool, default=True
    Whether to apply padding to the frequency (width) dimension of
    the input.
  bias : bool, default=True
    Whether to include a bias term in the convolutional layer.
  separable : bool, default=False
    Whether to use a separable convolution.
  norm_layer : callable, default=torch.nn.BatchNorm2d
    A callable that returns a normalization layer to apply after
    the convolution.
  activation_layer : callable, default=torch.nn.ReLU
    A callable that returns an activation layer to
    apply after normalization.
  trans_conv_type : str, default="conv_transpose"
    The type of transposed convolution to use. Options are
    "conv_transpose" and "up_sample".

  """

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      kernel_size: Union[int, Tuple[int, int]],
      fstride: int = 1,
      dilation: int = 1,
      fpad: bool = True,
      bias: bool = True,
      separable: bool = False,
      norm_layer: Optional[Callable[..., torch.nn.Module]
                           ] = torch.nn.BatchNorm2d,
      activation_layer: Optional[Callable[...,
                                          torch.nn.Module]] = torch.nn.ReLU,
      trans_conv_type: str = "conv_transpose",
  ):
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.kernel_size = kernel_size
    self.fstride = fstride
    self.dilation = dilation
    self.bias = bias
    self.trans_conv_type = trans_conv_type

    # Padding on time axis, with lookahead = 0
    lookahead = 0  # This needs to be handled on the input feature side
    kernel_size = (kernel_size, kernel_size) if isinstance(
        kernel_size, int) else kernel_size
    if fpad:
      fpad_ = kernel_size[1] // 2
    else:
      fpad_ = 0
    self.fpad = fpad_

    pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
    self.layers = []
    if any(x > 0 for x in pad):
      self.layers.append(nn.ConstantPad2d(pad, 0.0))
    groups = math.gcd(in_ch, out_ch) if separable else 1
    self.groups = groups
    if groups == 1:
      separable = False
    if trans_conv_type == "conv_transpose":
      self.layers.append(
          nn.ConvTranspose2d(
              in_ch,
              out_ch,
              kernel_size=kernel_size,
              padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
              output_padding=(0, fpad_),
              stride=(1, fstride),  # Stride over time is always 1
              dilation=(1, dilation),
              groups=groups,
              bias=bias,
          )
      )
    else:
      self.layers.append(
          SeparatedTransposedConv2d(
              self.in_ch,
              self.out_ch,
              kernel_size=self.kernel_size,
              padding=(self.kernel_size[0] - 1,
                       self.fpad + self.dilation - 1),
              output_padding=(0, self.fpad),
              stride=(1, self.fstride),  # Stride over time is always 1
              groups=self.groups,
              bias=self.bias,
              consistent=False,
          ))

    if separable:
      self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
    if norm_layer is not None:
      self.layers.append(norm_layer(out_ch))
    if activation_layer is not None:
      self.layers.append(activation_layer())
    super().__init__(*self.layers)


class GroupedLinearEinsum(nn.Module):
  """Applies a linear transformation to the input tensor using
  grouped weights.

  Parameters
  ----------
  input_size : int
    The number of expected features in the input.
  hidden_size : int
    The number of output features.
  groups : int, default=1
    Number of groups to divide the weights and input tensor into.

  """

  def __init__(self, input_size: int, hidden_size: int, groups: int = 1):

    super().__init__()
    # self.weight: Tensor
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.groups = groups
    assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
    assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"
    self.ws = input_size // groups
    self.register_parameter(
        "weight",
        Parameter(
            torch.zeros(groups, input_size // groups, hidden_size // groups),
            requires_grad=True),
    )
    self.reset_parameters()

  def reset_parameters(self):
    """Resets the weights."""
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

  def forward(self, x: Tensor) -> Tensor:
    """Applies the grouped linear einsum transformation to the input
    tensor.

    Parameters
    ----------
    x : torch.Tensor
      The input tensor of shape [B, T, input_size].

    Returns
    -------
    torch.Tensor
      The output tensor of shape [B, T, hidden_size].

    """
    # x: [..., I]
    b, t, _ = x.shape
    # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
    new_shape = (b, t, self.groups, self.ws)
    x = x.view(new_shape)
    # The better way, but not supported by torchscript
    # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
    x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
    x = x.flatten(2, 3)  # [B, T, H]
    return x


class SqueezedRNN_S(nn.Module):

  """A PyTorch module that implements a squeezed GRU, which is a variant
  of GRU with a smaller number of parameters.

  Parameters
  ----------
  input_size : int
    The number of expected features in the input tensor.
  hidden_size : int
    The number of features in the hidden state tensor.
  output_size : int, default=None
    The number of expected features in the output tensor.
    If None, the identity function is used as the output layer
  linear_groups : int, default=8
    The number of groups to use for the grouped linear layers.
  batch_first : bool, default=True
    If True, then the input and output tensors are provided
    as (batch, seq, feature).
  gru_skip_op : Callable[..., torch.nn.Module], default=None
    A callable function to apply as a skip connection.
    The default value of None means that no skip connection is used.
  linear_act_layer : Callable[..., torch.nn.Module], default=nn.Identity
    A callable function to use as an activation function for the linear
    layers.


  Methods
  -------
  forward(input, h=None)
    Perform the forward pass of the SqueezedGRU module.

  """

  def __init__(
      self,
      rnn_type: str,
      input_size: int,
      hidden_size: int,
      output_size: Optional[int] = None,
      num_layers: int = 1,
      linear_groups: int = 8,
      batch_first: bool = True,
      rnn_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
      linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
  ):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.linear_in = nn.Sequential(
        GroupedLinearEinsum(input_size, hidden_size,
                            linear_groups), linear_act_layer()
    )
    if rnn_type == "LiGRU":
      self.rnn = OptimizedLightGRU(hidden_size, hidden_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first)
    else:
      self.rnn = getattr(nn, rnn_type.upper())(hidden_size, hidden_size,
                                               num_layers=num_layers,
                                               batch_first=batch_first)

    self.rnn_skip = rnn_skip_op() if rnn_skip_op is not None else None
    if output_size is not None:
      self.linear_out = nn.Sequential(
          GroupedLinearEinsum(hidden_size, output_size,
                              linear_groups), linear_act_layer()
      )
    else:
      self.linear_out = nn.Identity()

  def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
    x = self.linear_in(input)
    x, h = self.rnn(x, h)
    x = self.linear_out(x)
    if self.rnn_skip is not None:
      x = x + self.rnn_skip(input)
    return x, h


class BaseModel(nn.Module):
  """Abstract class for all models.

  Methods
  -------
  from_pretrained(path)
    Load a trained model from a given path.
  from_checkpoint(path)
    Load a trained model from a checkpoint given path.
  """

  def __init__(self):
    super().__init__()
    self.name = "base_model"

  @classmethod
  def from_pretrained(cls, path, *args):
    """Load a trained model from a given path.

    Parameters
    ----------
    path : str
      The path to the saved model checkpoint.

    Returns
    -------
    Model
      A new instance of the `Model` class with the same architecture
      and parameters as the trained model checkpoint.
    """
    if os.path.isdir(path):
      path = os.path.join(path, "best_model.pth")
    state = torch.load(path, map_location="cpu")
    state_dict = state["state_dict"]
    key_to_remove = cls.key_to_remove()
    for key in list(state["config"].keys()):
      if key in key_to_remove:
        state["config"].pop(key)
    model = cls(*args, **state["config"])
    model.load_state_dict(state_dict)
    return model

  @classmethod
  def from_checkpoint(cls, path, *args):
    """Load a trained model from a given path.

    Parameters
    ----------
    path : str
      The path to the saved model checkpoint.

    Returns
    -------
    Model
      A new instance of the `Model` class with the same architecture
      and parameters as the trained model checkpoint.

    Note
    ----
    The difference between this method and `from_pretrained` is that
    this method loads the model from a checkpoint saved by the
    `Trainer` class that also contains the state of the optimizer,
    callbacks, etc..., while `from_pretrained` loads the model from an
    isolated version of this checkpoint that only contains the model's
    state dict and config.
    """
    state = torch.load(path, map_location="cpu")
    state_dict = state["state_dict"]
    key_to_remove = cls.key_to_remove()
    for key in list(state["training_config"]["model"].keys()):
      if key in key_to_remove:
        state["training_config"]["model"].pop(key)
    model = cls(*args, **state["training_config"]["model"])
    new_state = {}
    for key in state_dict.keys():
      new_state[key.replace("model.", "")] = state_dict[key]
    model.load_state_dict(new_state)
    return model

  @classmethod
  def key_to_remove(cls):
    """Get the key to remove from the state dict when loading a model"""
    return []
  
  
class Encoder(nn.Module):
  """
  Encoder module.

  Parameters
  ----------
  conv_ch : int, default=64
    Number of channels used in the convolutional layers (default is 64).
  conv_kernel_inp : tuple[int], default=(3,3)
    Kernel size for the initial convolutional layer that processes the
    input waveform (default is (3, 3)).
  conv_kernel : tuple[int], default=1,3)
    Kernel size for the convolutional layers that follow the initial
    layer, (default is (1, 3)).
  nb_erb : int, default=32
    Number of equivalent rectangular bandwidth (ERB) bins to use
    (default is 32).
  nb_df : int, default=96
    Number of gammatone filterbank (GTF) channels, by default 96.
  emb_hidden_dim : int, default=256
    Number of units in each layer of the encoder's bidirectional GRU
    (default is 256).
  emb_num_layers : int, default=2
    Number of layers in the encoder's bidirectional GRU (default is 2).
  lin_groups : int, default=96
    Number of groups to use in the grouped linear layers
    (default is 96).

  Attributes
  ----------
  erb_conv0 : Conv2dNormAct
    Initial convolutional layer that processes the input waveform.
  erb_conv1, erb_conv2, erb_conv3, df_conv0, df_conv1 : Conv2dNormAct
    Convolutional layers used to compress the waveform representation.
  df_fc_emb : nn.Sequential
    Grouped linear layer followed by a ReLU activation used to
    transform the representation obtained from the GTF filterbank.
  emb_in_dim, emb_out_dim, emb_n_layers : int
    Dimensions and number of layers in the encoder's bidirectional GRU.
  emb_gru : SqueezedGRU
    GRU used to compress the waveform representation.

  Methods
  -------
  forward(waveform)
    Applies the encoder to a noisy waveform to obtain a compressed
    representation.
  """

  def __init__(
      self,
      conv_ch: int = 64,
      conv_kernel_inp: (int) = (3, 3),
      conv_kernel: (int) = (1, 3),
      nb_erb: int = 24,
      nb_df: int = 96,
      emb_hidden_dim: int = 256,
      emb_num_layers: int = 1,
      lin_groups: int = 96,
      enc_lin_groups: int = 32,
      rnn_type="gru",
      branch=["erb", "df"],
      common_encoder=True,
  ):
    super().__init__()
    
    
    self.branch = branch
    self.branch_erb = False
    self.branch_df = False
    if "erb" in self.branch:
      self.branch_erb = True
    if "df" in self.branch:
      self.branch_df = True
      
    self.common_encoder = common_encoder
    
    assert self.branch_erb or self.branch_df, "At least one branch should be used"
    
    self.erb_bins = nb_erb
    self.emb_in_dim = conv_ch * (nb_erb // 4)
    self.emb_dim = emb_hidden_dim
    self.emb_out_dim = conv_ch * (nb_erb // 4)
    
    conv_layer = partial(
        Conv2dNormAct,
        in_ch=conv_ch,
        out_ch=conv_ch,
        kernel_size=conv_kernel,
        bias=False,
        separable=True,
    )    
    
    if self.common_encoder or self.branch_erb:
    
      self.erb_conv0 = Conv2dNormAct(
          1, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True
      )

      self.erb_conv1 = conv_layer(fstride=2)
      self.erb_conv2 = conv_layer(fstride=2)
      self.erb_conv3 = conv_layer(fstride=1)
      
    if self.common_encoder or self.branch_df:
      self.df_conv0 = Conv2dNormAct(
          2, conv_ch, kernel_size=conv_kernel_inp, bias=False, separable=True
      )
      self.df_conv1 = conv_layer(fstride=2)
      df_fc_emb = GroupedLinearEinsum(
          conv_ch * (nb_df // 2), self.emb_in_dim, groups=enc_lin_groups
      )
      self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
      
    
    self.emb_gru = SqueezedRNN_S(
        rnn_type,
        self.emb_in_dim,
        self.emb_dim,
        output_size=self.emb_out_dim,
        num_layers=emb_num_layers,
        batch_first=True,
        rnn_skip_op=None,
        linear_groups=lin_groups,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )

  def forward(
      self, feat_erb: Tensor, feat_spec: Tensor
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Forward pass of the Encoder module.

    Parameters
    ----------
    feat_erb : torch.Tensor
      The input tensor of the ERB spectrogram with shape [B, 1, T, Fe].
    feat_spec : torch.Tensor
      The input tensor of the complex spectrogram with shape [B, 2, T, Fc].

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor]
      A tuple of tensors containing the following:
      - e0: the output tensor of the first convolution layer on the
      ERB spectrogram with shape [B, C, T, F].
      - e1: the output tensor of the second convolution layer on the
      ERB spectrogram with shape [B, C*2, T, F/2].
      - e2: the output tensor of the third convolution layer on the
      ERB spectrogram with shape [B, C*4, T, F/4].
      - e3: the output tensor of the fourth convolution layer on the
      ERB spectrogram with shape [B, C*4, T, F/4].
      - emb: the output tensor of the GRU layer with shape [B, T, -1].
      - c0: the output tensor of the first convolution layer on the
      complex spectrogram with shape [B, C, T, Fc].
      - lsnr: the output tensor of the sigmoid layer for loudness
      estimation with shape [B, T, 1].
    """

    # Encodes erb; erb should be in dB scale + normalized;
    # Fe are number of erb bands.
    # erb: [B, 1, T, Fe]
    # spec: [B, 2, T, Fc]
    # b, _, t, _ = feat_erb.shape
    if (self.branch_erb and self.branch_df) or self.common_encoder:
      e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
      e1 = self.erb_conv1(e0)  # [B, C, T, F/2]
      e2 = self.erb_conv2(e1)  # [B, C, T, F/4]
      e3 = self.erb_conv3(e2)  # [B, C, T, F/4]
      
      c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
      c1 = self.df_conv1(c0)  # [B, C, T, Fc/2]
      cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
      cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
      
      emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F/4]
      emb += cemb
      emb, _ = self.emb_gru(emb)  # [B, T, -1]
      return e0, e1, e2, e3, emb, c0
    
    elif not self.common_encoder: 
      if self.branch_erb and not self.branch_df:
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]      
        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F/4]
        emb, _ = self.emb_gru(emb)
        return e0, e1, e2, e3, emb
      
      elif self.branch_df and not self.branch_erb:
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc]
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb, _ = self.emb_gru(cemb)
        return c0, emb
    
    
class ErbDecoder(nn.Module):
  """A neural network module that decodes the features extracted by the  
  ERB part of the `Encoder` module.

  Parameters
  ----------
  conv_ch : int
    Number of channels in convolution layers.
  conv_kernel : int
    Kernel size of the convolution layers.
  nb_erb : int
    Number of ERB bands.
  emb_hidden_dim : int
    The number of features in the GRU's hidden state.
  emb_num_layers : int
    The number of layers in the GRU.
  lin_groups : int
    Number of groups for the linear convolution layers.

  Attributes
  ----------
  emb_out_dim : int
    The number of features in the GRU's output.

  Methods
  -------
  forward(feat_erb, feat_spec)
    Forward pass through the network.

  """

  def __init__(
      self,
      conv_ch: int = 16,
      conv_kernel: (int) = (1, 3),
      nb_erb: int = 32,
      emb_hidden_dim: int = 256,
      emb_num_layers: int = 2,
      lin_groups: int = 96,
      rnn_type="gru",
      trans_conv_type="conv_transpose"

  ):
    super().__init__()
    assert nb_erb % 8 == 0, "erb_bins should be divisible by 8"

    self.emb_in_dim = conv_ch * (nb_erb // 4)
    self.emb_dim = emb_hidden_dim
    self.emb_out_dim = conv_ch * (nb_erb // 4)

    self.emb_gru = SqueezedRNN_S(
        rnn_type,
        self.emb_in_dim,
        self.emb_dim,
        output_size=self.emb_out_dim,
        num_layers=emb_num_layers,
        batch_first=True,
        rnn_skip_op=None,
        linear_groups=lin_groups,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )

    tconv_layer = partial(
        ConvTranspose2dNormAct,
        kernel_size=conv_kernel,
        bias=False,
        separable=True,
        trans_conv_type=trans_conv_type,

    )
    conv_layer = partial(
        Conv2dNormAct,
        bias=False,
        separable=True,
    )
    # convt: TransposedConvolution, convp: Pathway (encoder to decoder)
    # convolutions
    self.conv3p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt3 = conv_layer(conv_ch, conv_ch, kernel_size=conv_kernel)
    self.conv2p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt2 = tconv_layer(conv_ch, conv_ch, fstride=2)
    self.conv1p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt1 = tconv_layer(conv_ch, conv_ch, fstride=2)
    self.conv0p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.conv0_out = conv_layer(
        conv_ch, 2, kernel_size=conv_kernel, activation_layer=nn.Sigmoid
    )

  def forward(self, emb, e3, e2, e1, e0) -> Tensor:
    """Calculates the erb mask estimate.

    Parameters
    ----------
    emb : torch.Tensor
      Input tensor with shape [B, T, C], where B is the batch size,
      T is the time steps, and C is the number of input channels.
    e3 : torch.Tensor
      Input tensor with shape [B, C*4, T, F/4].
    e2 : torch.Tensor
      Input tensor with shape [B, C*2, T, F/2].
    e1 : torch.Tensor
      Input tensor with shape [B, C, T, F].
    e0 : torch.Tensor
      Input tensor with shape [B, 1, T, F].

    Returns
    -------
    torch.Tensor
      The erb mask estimate with shape [B, 2, T, F], converted into
      2 masks [B, 1, T, F].
    """
    # Estimates erb mask
    b, _, t, f8 = e3.shape
    emb, _ = self.emb_gru(emb)
    emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C, T, F/4]
    e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, T, F/4]
    e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, T, F/2]
    e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
    m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 2, T, F]
    i_m = m[:, 0:1, :, :]
    s_m = m[:, 1:2, :, :]
    return i_m, s_m # [B, 1, T, F], [B, 1, T, F]
  
  
class DfDecoder(nn.Module):
  """A PyTorch module that decodes the learned representations into
  the predicted spectrogram.

  Parameters
  ----------
  conv_ch : int, default=64
    The number of output channels for the convolutions (default is 64).
  nb_df : int, default=96
    The number of frequency bins for the decoder (default is 96).
  emb_hidden_dim : int, default=256
    The dimensionality of the embedding space (default is 256).
  lin_groups : int, default=96
    The number of groups for the linear layers (default is 96).
  df_hidden_dim : int, default=256
    The number of hidden units in each decoder GRU layer
    (default is 256).
  df_num_layers : int, default=2
    The number of GRU layers in the decoder (default is 2).
  df_order : int, default=5SqueezedLSTM_S
  """

  def __init__(
      self,
      conv_ch: int = 64,
      nb_df: int = 96,
      nb_erb: int = 32,
      lin_groups: int = 96,
      df_hidden_dim: int = 256,
      df_num_layers: int = 2,
      df_order: int = 5,
      df_lookahead: int = 2,
      gru_groups: int = 1,
      df_pathway_kernel_size_t: int = 1,
      rnn_type="gru"
  ):
    super().__init__()
    layer_width = conv_ch

    self.emb_in_dim = conv_ch * nb_erb // 4
    self.emb_dim = df_hidden_dim
    self.df_n_hidden = df_hidden_dim
    self.df_n_layers = df_num_layers
    self.df_order = df_order
    self.df_bins = nb_df
    self.df_lookahead = df_lookahead
    self.gru_groups = gru_groups
    self.df_out_ch = df_order * 2

    kt = df_pathway_kernel_size_t
    self.df_convp = Conv2dNormAct(
        layer_width, 2 * self.df_out_ch, fstride=1, kernel_size=(kt, 1),
        separable=True, bias=False
    ) # [B, C, T, F] -> [B, O*4, T, F]

    self.df_gru = SqueezedRNN_S(
        rnn_type,
        self.emb_in_dim,
        df_hidden_dim,
        num_layers=self.df_n_layers,
        batch_first=True,
        rnn_skip_op=None,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )

    out_dim = self.df_bins * (2 * self.df_out_ch)
    self.df_skip = GroupedLinearEinsum(
        self.emb_in_dim, self.emb_dim, groups=lin_groups)
    df_out = GroupedLinearEinsum(self.df_n_hidden, out_dim, groups=lin_groups)
    self.df_out = nn.Sequential(df_out, nn.Tanh())

  def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:

    b, t, _ = emb.shape
    c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
    c = c + self.df_skip(emb)  # [B, T, H]
    c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*4], channels_last
    c = self.df_out(c)  # [B, T, F*O*4], O: df_order
    c = c.view(b, t, self.df_bins, 2*self.df_out_ch) + c0  # [B, T, F, O*4]
    c = c.view(b, t, self.df_bins, self.df_out_ch, -1)       # [B, T, F, 0*2, 2]
    i_c = c[:,:,:,:,0]
    s_c = c[:,:,:,:,1]
    return i_c, s_c
  
  
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
  
  
def freq2erb(freq_hz: float):
  """Converts frequency in Hertz to ERB scale.

  Parameters
  ----------
  freq_hz : float
    The frequency in Hz.

  Returns
  -------
  float
    The frequency on ERB scale.

  """
  freq_erb = 9.265 * torch.log(1 + freq_hz / (24.7 * 9.265))
  return freq_erb


def erb2freq(freq_erb: float):
  """Converts ERB scale to frequency in Hertz.

  Parameters
  ----------
  freq_erb : float
    The frequency on the ERB scale.

  Returns
  -------
  float
    The frequency in Hz.

  """
  freq_hz = (torch.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
  return freq_hz


class ERBFB(nn.Module):
  """Equivalent Rectangular filterbank.

  Parameters
  ----------
  sr : int
    Sampling rate of the inputs.
  fft_size : int
    The used FFT size.
  n_bands : int
    The number of bands to compute.
  min_n_freqs : int, default=2
    The min number of spectral frequency in one band.
  normalized : bool, default=True,
    Wheter to normalize or not.
  min_mean_norm : int, default=-60
    Start point of mean_norm array.
  max_mean_norm : int, default=-90
    Last point of mean_norm array.
  alpha : float, default=0.99
    Multiplier of the moving average normalization.

  Methods
  -------
  generate_bands()
    Generates the bands of frequencies.
  erb_fb(normalized=True,inverse=False)
    Computes the filterbank tensor.
  init_mean_norm_state()
    Instantiate mean_norm_state.
  compute_band_corr(c, p)
    Computes correlation of spec frequencies associated to the same band.
  band_mean_norm_erb(x)
    Performs Exponential mean average over x.
  forward(spec)
    Computes the ERB normalized features from a spec.
  inverse(erb_feat)
    Return spec gains from ERB features.

  Examples
  --------
  >>> import torch
  >>> from speech_enhancement.dsp.erb import ERBFB
  >>> from speech_enhancement.dsp.stft import STFTFB
  >>> x = torch.randn(2, 16000)
  >>> spec = STFTFB(512, 512, 256)(x)
  >>> erb_spec = ERBFB(16000, 512, 128)(spec)
  """

  def __init__(
      self, sr: int, fft_size: int, n_bands: int, min_n_freqs=2,
      normalized=True, min_mean_norm=-60, max_mean_norm=-90, alpha=0.99
  ):
    super().__init__()
    self.sr = sr
    self.fft_size = fft_size
    self.n_bands = n_bands
    self.nyq_freq = sr / 2
    self.freq_width = sr / fft_size
    self.alpha = alpha
    self.erb_low = freq2erb(torch.tensor(0.0))
    self.erb_high = freq2erb(torch.tensor(self.nyq_freq))
    self.n_freq_bands_by_erb_bands = torch.zeros(n_bands)
    self.step = (self.erb_high - self.erb_low) / n_bands
    self.min_nb_freqs = min_n_freqs
    self.prev_freq = 0
    self.freq_over = 0
    self.n_freq_bands_by_erb_bands = self.generate_bands()
    self.min_mean_norm = min_mean_norm
    self.max_mean_norm = max_mean_norm
    self.mean_norm_state = self.init_mean_norm_state()
    self.normalized = normalized

  def generate_bands(self):
    """ 
    Generates the bands of frequencies. The spec [B,F,T] is divided
    into # bands, the width of each band is stored in
    self.n_freq_bands_by_erb_bands
    sum(self.n_freq_bands_by_erb_bands) == F

    Returns
    -------
    Tensor
      Tensor that maps each frequency of the spec to its band.

    """
    for i in range(1, self.n_bands + 1):
      f = erb2freq(self.erb_low + i * self.step)
      fb = (f / self.freq_width).round()
      nb_freqs = fb - self.prev_freq - self.freq_over
      if nb_freqs < self.min_nb_freqs:
        self.freq_over = self.min_nb_freqs - nb_freqs
        nb_freqs = self.min_nb_freqs
      else:
        self.freq_over = 0
      self.n_freq_bands_by_erb_bands[i - 1] = nb_freqs
      self.prev_freq = fb
    self.n_freq_bands_by_erb_bands[self.n_bands - 1] += 1
    too_large = sum(self.n_freq_bands_by_erb_bands) - (self.fft_size / 2 + 1)
    if too_large > 0:
      self.n_freq_bands_by_erb_bands[self.n_bands - 1] -= too_large
    return self.n_freq_bands_by_erb_bands

  def erb_fb(
      self,
      normalized: bool = True,
      inverse: bool = False,
  ) -> Tensor:
    """
    Computes the filterbank tensor.

    Parameters
    ----------
    normalized : bool, default=True
      Wheter to normalize or not.
    inverse : bool, default=False
      Wheter to return the inverse fb or not.

    Returns
    -------
    Tensor
      The filter bank to be mutiply with spec. Shape [F, n_bands].
      or its transpose if inverse=True.

    """

    widths = self.n_freq_bands_by_erb_bands.data.int()  # .numpy().astype(int)
    n_freqs = int(sum(widths))
    all_freqs = torch.linspace(0, self.sr // 2, n_freqs + 1)[:-1]
    b_pts = torch.cumsum(
        torch.cat((torch.tensor([0]), widths)), dim=0).int()[:-1]
    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
      fb[b: b + w, i] = 1
    # Normalize to constant energy per resulting band
    if inverse:
      fb = fb.t()
      if not normalized:
        fb /= fb.sum(dim=1, keepdim=True)
    else:
      if normalized:
        fb /= fb.sum(dim=0)
    return fb

  def init_mean_norm_state(self):
    """Instantiate mean_norm_state with the provided min and max values.

    Returns
    -------
    Tensor
      The tensor linearly spaced between min and max. Shape [n_bands].

    """
    return torch.linspace(self.min_mean_norm, self.max_mean_norm,
                          len(self.n_freq_bands_by_erb_bands))

  def band_mean_norm_erb(self, x):
    """
    Performs Exponential mean average over x.

    Parameters
    ----------
    x : Tensor
      ERB features with shape [B, 1, n_bands, T].

    Returns
    -------
    Tensor
      ERB features normalized with shape [B, 1, n_bands, T].

    """
    # x.shape = [B,1,self.n_bands,T]
    x = x.squeeze(1)
    s = self.init_mean_norm_state().to(x.device)
    for i in range(x.shape[2]):
      s = x[:, :, i] * (1.0 - self.alpha) + s * self.alpha
      x[:, :, i] -= s
      x[:, :, i] /= 40.0
    return x.unsqueeze(1)

  def forward(self, spec):
    """Computes the ERB normalized features from a spec.

    Parameters
    ----------
    spec : Tensor
      Complex valued spectrogram with shape [B, 1, F, T].

    Returns
    -------
    Tensor
      ERB normalized features (real valued) with shape [B, 1, n_bands, T].

    """
    # spec (complex) [B,1,F,T]
    fb = self.erb_fb().to(spec.device)
    spec_abs = spec.abs().square()
    erb = spec_abs.transpose(-1, -2).matmul(fb).transpose(-1, -2)
    erb = 10 * torch.log10(erb + 1e-10)
    if not self.normalized:
      return erb
    erb_normalized = self.band_mean_norm_erb(erb)
    return erb_normalized  # erb (real) [B,1,n_bands,T]  
  
  
class FeatureSpecExtractor(nn.Module):
  """A PyTorch module that performs feature extraction on a complex
  spectrogram.

  Parameters
  ----------
  n_feat : int
    The number of frequency bands to extract from the spectrogram.
  alpha : float
    A coefficient that controls the trade-off between the current and
    previous norms of the extracted features.
  normalized : bool, default=True
    Whether to normalize the extracted features.
  min_unit_norm : float, default=0.001
    The minimum value for the unit norm of the extracted features.
  max_unit_norm : float, default=0.0001
    The maximum value for the unit norm of the extracted features.

  Methods
  -------
  init_unit_norm_state()
    Initializes the `unit_norm_state` tensor with a linearly spaced
    sequence of values between `min_unit_norm` and `max_unit_norm`.
  band_unit_norm(x, s, alpha)
    Applies band-wise unit normalization to the input tensor `x`,
    using the scaling factor `s` and the coefficient `alpha`.
  forward(spec)
    Performs feature extraction on the input spectrogram `spec` and
    returns the normalized feature tensor.

  """

  def __init__(self, n_feat, alpha, normalized=True, min_unit_norm=0.001,
               max_unit_norm=0.0001) -> None:
    super().__init__()
    self.n_feat = n_feat
    self.alpha = alpha
    self.min_unit_norm = min_unit_norm
    self.max_unit_norm = max_unit_norm
    self.unit_norm_state = self.init_unit_norm_state()
    self.normalized = normalized

  def init_unit_norm_state(self):
    """Initializes the `unit_norm_state` tensor with a linearly spaced
    sequence of values between `min_unit_norm` and `max_unit_norm`.

    Returns
    -------
    Tensor
      A tensor of shape [n_feat] that stores the current unit norm
      state of the extracted features.
    """
    return torch.linspace(self.min_unit_norm, self.max_unit_norm, self.n_feat)

  def band_unit_norm(self, x, s, alpha):
    """Applies band-wise unit normalization to the input tensor `x`,
    using the scaling factor `s` and the coefficient `alpha`.

    Parameters
    ----------
    x : Tensor
      The input tensor of shape [B, C, n_feat, T].
    s : Tensor
      The scaling factor tensor of shape [n_feat].
    alpha : float
      A coefficient that controls the trade-off between the current
      and previous norms of the extracted features.

    Returns
    -------
    Tensor
      The normalized output tensor of shape [B, C, n_feat, T].
    """
 
    for i in range(x.shape[-1]):
      s = torch.abs(x[..., i]) * (1.0 - alpha) + s * alpha
      x[..., i] /= torch.sqrt(s)
    return x

  def forward(self, spec):
    """
    Performs feature extraction on the input spectrogram `spec`
    and returns the normalized feature tensor.

    Parameters
    ----------
    spec : Tensor
      The input complex spectrogram tensor of shape [B, C, F, T]`.

    Returns
    -------
    Tensor
        The output tensor of shape [B, C, n_feat, T].
    """
    # spec (complex) [B,1,F,T]
    # Need to clone otherwise spec is modified
    extracted_spec = spec.squeeze(1)[..., :self.n_feat, :].clone()

    if not self.normalized:
      return extracted_spec.unsqueeze(1)
    # return (complex) [B,1,n_feat,T]
    return self.band_unit_norm(extracted_spec,
                               self.unit_norm_state.to(spec.device),
                               self.alpha).unsqueeze(1) 
  
class Mask(nn.Module):
  """
  Applies a mask to a spectrogram.

  Used to apply the ERB gains to the spectrogram.

  Parameters
  ----------
  erb_inv_fb : torch.Tensor
    The inverse gammatone filterbank matrix with shape [F, Fe], where F
    is the number of frequency bins in the spectrogram and Fe is the number
    of ERB frequency bins.
  post_filter : bool, default=False
    Whether to apply a post-filter to the mask.  
  eps : float, default=1e-12
    A small value used to avoid division by zero when computing the
    post-filter.

  Methods
  -------
  forward(spec, mask, atten_lim=None)
    Applies the mask to the input spectrogram.

  pf(mask, beta=0.02)
    Applies a post-filter to the mask.

  """

  def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):

    super().__init__()
    self.erb_inv_fb: Tensor
    self.register_buffer("erb_inv_fb", erb_inv_fb)
    self.clamp_tensor = torch.__version__ > "1.9.0" or torch.__version__ == "1.9.0"
    self.post_filter = post_filter
    self.eps = eps

  def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
    """Applies a post-filter to the input mask.

    Parameters
    ----------
    mask : torch.Tensor
      The input mask tensor with shape [B, 1, T, Fe].
    beta : float, default=0.02
      The post-filter parameter.

    Returns
    -------
    torch.Tensor
      The post-filtered mask tensor with shape [B, 1, T, Fe].

    """
    pi = 3.14159265359
    mask_sin = mask * torch.sin(pi * mask / 2)
    mask_pf = (1 + beta) * mask / (1 + beta *
                                   mask.div(mask_sin.clamp_min(self.eps)).pow(2))
    return mask_pf

  def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
    """Applies the mask to the input spectrogram.

    Parameters
    ----------
    spec : torch.Tensor
      The input spectrogram tensor with shape [B, 1, T, F, 2], where F 
      is the number of frequency bins.
    mask : torch.Tensor
      The input mask tensor with shape [B, 1, T, Fe], where Fe is the
      number of ERB frequency bins.
    atten_lim : torch.Tensor, default=None
      The attenuation limit tensor with shape [B], in dB.

    Returns
    -------
    torch.Tensor
      The masked spectrogram tensor with shape [B, 1, T, F, 2].

    """

    # spec (real) [B, 1, T, F, 2], F: freq_bins
    # mask (real): [B, 1, T, Fe], Fe: erb_bins
    # atten_lim: [B]
    if not self.training and self.post_filter:
      mask = self.pf(mask)
    if atten_lim is not None:
      # dB to amplitude
      atten_lim = 10 ** (-atten_lim / 20)
      # Greater equal (__ge__) not implemented for TorchVersion.
      if self.clamp_tensor:
        # Supported by torch >= 1.9
        mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
      else:
        m_out = []
        for i in range(atten_lim.shape[0]):
          m_out.append(mask[i].clamp_min(atten_lim[i].item()))
        mask = torch.stack(m_out, dim=0)
    mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
    return spec * mask.unsqueeze(4)
  
  
class DfOutputReshapeMF(nn.Module):
  """
  Reshape the input tensor from [B, T, F, O*2] to [B, O, T, F, 2].

  Methods
  -------
  forward(coefs)
    Reshape the input tensor.

  """

  def __init__(self):
    super().__init__()

  def forward(self, coefs: Tensor) -> Tensor:
    """Reshape the input tensor.

    Parameters
    ----------
    coefs : torch.Tensor
      The input tensor to be reshaped. It has shape [B, T, F, O*2].

    Returns
    -------
    torch.Tensor
      The reshaped tensor, with shape [B, O, T, F, 2].

    """
    # [B, T, F, O*2] -> [B, O, T, F, 2]
    coefs = coefs.unflatten(-1, (-1, 2)).permute(0, 3, 1, 2, 4)
    return coefs

  
class ImpulseSoundRejection(BaseModel):

  def __init__(
      self,
      conv_ch: int = 64,
      conv_kernel_inp: (int) = (1, 3),
      conv_kernel: (int) = (1, 3),
      conv_lookahead: int = 0,
      nb_erb: int = 32,
      nb_df: int = 256,
      emb_hidden_dim: int = 256,
      enc_num_layers: int = 1,
      lin_groups: int = 16,
      enc_linear_groups: int = 32,
      rnn_type="gru",
      trans_conv_type="conv_transpose",
      erb_num_layers: int = 2,
      df_hidden_dim: int = 256,
      df_num_layers: int = 2,
      df_order: int = 5,
      df_lookahead: int = 0,
      gru_groups: int = 1,
      df_pathway_kernel_size_t: int = 5,
      mask_pf: bool = False,
      df_n_iter: int = 1,
      stft_params=None,
      feat_spec_params=None,
      erb_params=None,
      erb_norm=None,
      feat_spec_norm=None,
      branch=["erb", "df"],
      common_encoder: bool = True,
      **kwargs
  ):
    """A deep learning model for speech enhancement.

    Parameters
    ----------
    conv_ch : int, default=64
      The number of convolution channels (default is 64).
    nb_df : int, default=96
      The number of decorrelated feature channels (default is 96).
    emb_hidden_dim : int, default=256
      The hidden dimension of the embedding layer (default is 256).
    lin_groups : int, default=8
      The number of linear groups (default is 8.
    df_hidden_dim : int, default=256
      The hidden dimension of the DF decoder (default is 256).
    df_num_layers : int, default=2
      The number of layers in the DF decoder (default is 2).
    df_order : int, default=5
      The order of the decorrelation filter (default is 5).
    df_lookahead : int, default=2
      The lookahead of the decorrelation filter (default is 2).
    gru_groups : int, default=1
      The number of GRU groups (default is 1).
    df_pathway_kernel_size_t : int, default=5
      The kernel size of the DF pathway (default is 5).
    conv_kernel_inp : int, tuple, default=(3,3)
      The input convolution kernel size (default is (3, 3)).
    conv_kernel : int, tuple, default=(1,3)
      The convolution kernel size (default is (1, 3)).
    nb_erb : int, default=32
      The number of ERB bins (default is 32).
    emb_num_layers : int, default=3
      The number of layers in the embedding layer (default is 3).
    lsnr_max : int, default=35
      The maximum signal-to-noise ratio (default is 35).
    lsnr_min : int, default=-15
      The minimum signal-to-noise ratio (default is -15).
    conv_lookahead : int, default=2
      The convolution lookahead (default is 2).
    fft_size : int, default=960
      The FFT size (default is 960).
    mask_pf : bool, default=False
      Whether to apply a post-filter to the mask (default is False).
    df_n_iter : int, default=1
      The number of iterations in the decorrelation filter
      (default is 1).
    stft_params : dict, default=None
      Parameters for the STFTFB class.
    feat_spec_params : dict, default=None
      Parameters for the FeatureSpecExtractor class.
    erb_params : dict, default=None
      Parameters for the ERBFB class.
      
    Attributes
    ----------
    name : str
      The name of the model.
    
    Methods
    -------
    forward(spec, feat_erb, feat_spec)
      Forward pass of the model.

    """

    super().__init__()
    self.name = "ImpulseSoundRejection"
    
    
    self.branch = branch
    self.branch_erb = False
    self.branch_df = False
    if "erb" in self.branch:
      self.branch_erb = True
    if "df" in self.branch:
      self.branch_df = True    
      
    self.common_encoder = common_encoder
    
    if self.branch_erb and self.branch_df:
      assert self.common_encoder, "Common encoder must be True if both branches are used."

    self.stft_params = stft_params
    self.erb_params = erb_params
    self.feat_spec_params = feat_spec_params

    if stft_params is not None:
      self.stft = STFTFB(**stft_params)
    if feat_spec_params is not None:
      self.feat_spec_extractor = FeatureSpecExtractor(**feat_spec_params)
    if erb_params is not None:
      self.erb = ERBFB(**erb_params)
      self.erb_fb = self.erb.erb_fb()
      self.erb_inv_fb = self.erb.erb_fb(inverse=True)

    layer_width = conv_ch
    self.nb_df = nb_df
    self.lookahead = conv_lookahead
    self.emb_dim = layer_width * nb_erb
    self.erb_bins = nb_erb
    self.rnn_type = rnn_type
    self.enc_num_layers = enc_num_layers
    self.emb_hidden_dim = emb_hidden_dim
    self.erb_num_layers = erb_num_layers
    self.conv_ch = conv_ch
    self.trans_conv_type = trans_conv_type
    
    if self.branch_df:
      self.df_pathway_kernel_size_t = df_pathway_kernel_size_t
      self.df_order = df_order
      self.nb_df = nb_df
      self.df_lookahead = df_lookahead
      self.df_num_layers = df_num_layers
      self.df_n_iter = df_n_iter

    if conv_lookahead > 0:
      pad = (0, 0, -conv_lookahead, conv_lookahead)
      self.pad = nn.ConstantPad2d(pad, 0.0)
    else:
      self.pad = nn.Identity()
    self.enc = Encoder(conv_ch,
                       conv_kernel_inp,
                       conv_kernel,
                       nb_erb,
                       nb_df,
                       emb_hidden_dim,
                       enc_num_layers,
                       lin_groups,
                       enc_linear_groups,
                       rnn_type,
                       branch,
                       common_encoder,)
    if self.branch_erb:
      self.erb_dec = ErbDecoder(conv_ch,
                                conv_kernel,
                                nb_erb,
                                emb_hidden_dim,
                                erb_num_layers,
                                lin_groups,
                                rnn_type,
                                trans_conv_type
                                )
      self.mask = Mask(self.erb_inv_fb, post_filter=mask_pf)
      
    if erb_norm is None or not self.branch_erb:
      self.erb_norm = nn.Identity()
    else:
      self.erb_norm = nn.BatchNorm2d(num_features=1)      

    if self.branch_df:
      self.forward_df = ApplyDF(df_order, nb_df, df_lookahead)
      self.df_dec = DfDecoder(conv_ch,
                              nb_df,
                              nb_erb,
                              lin_groups,
                              df_hidden_dim,
                              df_num_layers,
                              df_order,
                              df_lookahead,
                              gru_groups,
                              df_pathway_kernel_size_t,
                              rnn_type
                              )
      self.df_out_transform = DfOutputReshapeMF()
      self.pad_df = nn.ConstantPad2d(
          (0, 0, df_order - 1 - df_lookahead, df_lookahead), 0.0)

    if feat_spec_norm is None or not self.branch_df:
      self.feat_spec_norm = nn.Identity()
    else:
      self.feat_spec_norm = nn.BatchNorm2d(num_features=2)

  def forward(
      self,
      spec, feat_erb, feat_spec
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward pass of the model.

    Parameters
    ----------
    spec: torch.Tensor
      The complex spectrogram of the mixed audio of
      shape (1,1, stft_size/2 +1 , time).
    feat_erb: torch.Tensor
      ERB features (real valued) extracted from the spectrogram of
      shape (1,1, nb_erb, time).
    feat_spec: torch.Tensor
      Complex spectral features extracted from the spectrogram of
      shape (1,1, nb_df, time).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor,]
      A tuple of output tensors including:
      - spec (torch.Tensor): The processed audio spectrogram with shape
      (1,1, stft_size/2 +1 , time).

    """

    spec = torch.view_as_real(spec.transpose(-2, -1))
    feat_erb = feat_erb.transpose(-1, 2)
    feat_spec = torch.view_as_real(feat_spec.transpose(-1, -2))
    feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2).contiguous()

    feat_erb = self.erb_norm(feat_erb)    # batch norm num_features=1 <- channel dim
    feat_spec = self.feat_spec_norm(feat_spec)  # batch norm num_features=2 ? <- channel dim

    feat_erb = self.pad(feat_erb)  # self.pad = Identity if conv_lookahead == 0
    feat_spec = self.pad(feat_spec)
    
    if self.common_encoder:
      e0, e1, e2, e3, emb, c0 = self.enc(feat_erb, feat_spec)
      
      if self.branch_erb:
        i_m, s_m = self.erb_dec(emb, e3, e2, e1, e0)
        # Inverse ERB transform
        i_spec_m = self.mask(spec, i_m)
        s_spec_m = self.mask(spec, s_m)  
      else: 
        i_spec_m = spec
        s_spec_m = spec
      if self.branch_df:
        # DF coeffs
        i_df_coefs, s_df_coefs = self.df_dec(emb, c0)
        i_df_coefs = self.df_out_transform(i_df_coefs).contiguous()
        s_df_coefs = self.df_out_transform(s_df_coefs).contiguous()      
        i_spec = self.forward_df(i_spec_m, i_df_coefs) # [B, 1, F, T, 2] (view as real)
        s_spec = self.forward_df(s_spec_m, s_df_coefs) # [B, 1, F, T, 2] (view as real)   
      else:
        i_spec = i_spec_m
        s_spec = s_spec_m   
          
    elif self.branch_erb and not self.branch_df:
      e0, e1, e2, e3, emb = self.enc(feat_erb, feat_spec)
      i_m, s_m = self.erb_dec(emb, e3, e2, e1, e0)
      # Inverse ERB transform
      i_spec = self.mask(spec, i_m)
      s_spec = self.mask(spec, s_m)        
      
    elif self.branch_df and not self.branch_erb:
      c0, emb = self.enc(feat_erb, feat_spec)
      # DF coeffs
      i_df_coefs, s_df_coefs = self.df_dec(emb, c0)
      i_df_coefs = self.df_out_transform(i_df_coefs).contiguous()
      s_df_coefs = self.df_out_transform(s_df_coefs).contiguous()        
      i_spec = self.forward_df(spec, i_df_coefs) # [B, 1, F, T, 2] (view as real)
      s_spec = self.forward_df(spec, s_df_coefs) # [B, 1, F, T, 2] (view as real)     
    
    return i_spec, s_spec

  def forward_audio(
      self,
      x: Tensor,
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward pass of the model.

    Parameters
    ----------
    x : torch.Tensor
      Input audio waveform tensor with shape (batch_size, audio_length).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor,]
      A tuple of output tensors including:
      - spec (torch.Tensor): The processed audio spectrogram with shape 
      (batch_size, freq_bins, frames).
      - lsnr (torch.Tensor): The log-spectral magnitude estimation 
      of the input audio with shape (batch_size, 1, frames).

    """

    spec, feat_erb, feat_spec = self.get_features(x)
    return self.forward(spec, feat_erb, feat_spec)

  def get_features(self, x):
    """Compute the features extracted by the model. 

    Parameters
    ----------
    x : torch.Tensor
      Input audio waveform tensor with shape (batch_size, audio_length).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
      A tuple of tensors containing the following:
      - spec: the spectrogram of the input audio with shape [B, 1, F, T]. 
      - feat_erb: the ERB spectrogram of the input audio with shape 
      [B, 1, n_erb, T]. 
      - feat_spec: the complex spectrogram of the input audio with shape 
      [B, 1, nb_df, T]

    """
    spec = self.stft(x)
    feat_erb = self.erb(spec)
    feat_spec = self.feat_spec_extractor(spec)
    return spec, feat_erb, feat_spec
  
  
  
  
  
def main():
  
  stft_params = {"nfft": 2048,
                  "overlap": 0.75,
                  "window_size": 2048,
                  "window": None,
                  "center": True
               }
  erb_params = {"sr": 44100, "fft_size": stft_params["nfft"], "n_bands": 32,
                "min_n_freqs": 2, "normalized": False, "min_mean_norm": -60,
                "max_mean_norm": -90, "alpha": 0.99}
  feat_spec_params = {"n_feat": 256, "alpha": 0.99, "normalized": False,
                      "min_unit_norm": 0.001, "max_unit_norm": 0.0001}
  
  erb_norm = True 
  feat_spec_norm = True
  
  model = ImpulseSoundRejection(
    stft_params=stft_params,
    erb_params=erb_params,
    feat_spec_params=feat_spec_params,
    erb_norm=erb_norm,
    feat_spec_norm=feat_spec_norm
  )
  
  # random mono signal
  x = torch.randn(1, 441000)
  spec, feat_erb, feat_spec = model.get_features(x)
  
  print('x :', x.shape)
  print('spec :', spec.shape)
  print('feat_erb :', feat_erb.shape)
  print('feat_spec :', feat_spec.shape)
  
  
  i_res, s_res = model.forward(spec, feat_erb, feat_spec)
  
  print('i_res :', i_res.shape)
  print('s_res :', s_res.shape)
  
  
  
  
  
  

if __name__ == "__main__":
  main()
  