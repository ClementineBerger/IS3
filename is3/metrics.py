""" 
Metrics for the model evaluation.
"""

import torch
import torch.nn.functional as F

def frame_signal(signal, frame_size):
    """
    Splits a batch signal [B, T] into frames of size `frame_size` without overlap.
    
    Args:
        signal (torch.Tensor): Tensor [B, T], input signal.
        frame_size (int): Size of each frame.
    
    Returns:
        torch.Tensor: Tensor [B, N, frame_size] with the frames.
    """
    B, T = signal.shape
    num_frames = T // frame_size  
    signal = signal[:, :num_frames * frame_size]  
    frames = signal.view(B, num_frames, frame_size)  
    return frames

def compute_si_sdr(estimated_signal, reference_signal, scaling=True, remove_silences=False, 
                   frame_size=1024, delta=1e-2):
    """
    Compute the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) for a batch of audio signals.
    
    Args:
        estimated_signal (torch.Tensor): Tensor [B, T], estimated signals.
        reference_signal (torch.Tensor): Tensor [B, T], reference signals.
        scaling (bool): If True, applies an optimal scaling factor.
        remove_silences (bool): If True, removes silent frames.
        frame_size (int): Frame size for silence removal.
        delta (float): Threshold as a percentage of the max RMS to filter silences.
    
    Returns:
        torch.Tensor: SI-SDR [B, 1] after silence removal (if enabled).
    """
    eps = 1e-12  # To avoid division by zero
    
    if remove_silences:
        # Split into non-overlapping frames
        ref_frames = frame_signal(reference_signal, frame_size)  # [B, N, frame_size]
        est_frames = frame_signal(estimated_signal, frame_size)  # [B, N, frame_size]

        # Compute RMS of the frames
        rms_ref = torch.sqrt(torch.mean(ref_frames**2, dim=2))  # [B, N]
        
        # Filtering threshold based on delta % of max RMS
        rms_max = torch.max(rms_ref, dim=1, keepdim=True).values  # [B, 1]
        mask = rms_ref > (delta * rms_max)  # [B, N] -> True if the frame is kept
        
        # Apply the mask: keep useful frames
        ref_frames_filtered = torch.where(mask.unsqueeze(-1), ref_frames, torch.tensor(0.0, device=reference_signal.device))
        est_frames_filtered = torch.where(mask.unsqueeze(-1), est_frames, torch.tensor(0.0, device=estimated_signal.device))
        
        # Reconstruct the filtered signal with padding
        reference_signal = ref_frames_filtered.reshape(reference_signal.shape[0], -1)
        estimated_signal = est_frames_filtered.reshape(estimated_signal.shape[0], -1)

    # If all frames were removed, return -inf (total silence)
    if reference_signal.shape[1] == 0:
        return torch.full((reference_signal.shape[0], 1), float('-inf'), device=reference_signal.device)

    # Compute the optimal scaling factor
    if scaling:
        scale = (eps + torch.sum(reference_signal * estimated_signal, dim=1, keepdim=True)) / \
                (eps + torch.sum(reference_signal**2, dim=1, keepdim=True))
    else:
        scale = 1.0

    # Project the estimated signal onto the reference
    e_true = scale * reference_signal

    # Residual noise
    e_res = estimated_signal - e_true

    # Powers
    signal_power = torch.sum(e_true**2, dim=1)
    noise_power = torch.sum(e_res**2, dim=1)

    # Compute SI-SDR
    si_sdr = 10 * torch.log10((eps + signal_power) / (eps + noise_power))

    return si_sdr  # Shape [B, 1]

  
  
  
  
import numpy as np

def compute_si_sdr_np(estimated_signal, reference_signal, scaling=True):
    """
    Compute the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) for a batch of audio signals in NumPy.
    
    Args:
        estimated_signal (np.ndarray): Array of shape [B, T], estimated signals.
        reference_signal (np.ndarray): Array of shape [B, T], reference signals.
        scaling (bool): Whether to apply scaling to the reference signal.
    
    Returns:
        np.ndarray: SI-SDR values for the batch, shape [B, 1].
    """
    # Small epsilon to avoid division by zero
    eps = 1e-12

    # Compute the scaling factor for the projection
    if scaling:
        scale = (eps + np.sum(reference_signal * estimated_signal, axis=1, keepdims=True)) / \
                (eps + np.sum(reference_signal**2, axis=1, keepdims=True))
    else:
        scale = 1.0

    # Compute the true source component (scaled reference signal)
    e_true = scale * reference_signal

    # Compute the distortion (residual between estimated signal and projection)
    e_res = estimated_signal - e_true

    # Compute powers
    signal_power = np.sum(e_true**2, axis=1)
    noise_power = np.sum(e_res**2, axis=1)

    # Compute SI-SDR
    si_sdr = 10 * np.log10((eps + signal_power) / (eps + noise_power))

    return si_sdr  # Shape [B,]
