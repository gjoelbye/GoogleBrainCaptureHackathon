from typing import Tuple
import torch
import numpy as np
from torch import Tensor, nn
from math import ceil
from copy import deepcopy

def _make_span_from_seeds(seeds: np.ndarray, span: int, total: None | int = None) -> np.ndarray:
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape: Tuple[int, int], prob: float, total: int, span: int, allow_no_inds: bool = False) -> torch.Tensor:
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and prob > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < prob)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

class Permute(nn.Module):
    def __init__(self, *axes: int) -> None:
        super().__init__()
        self.axes = axes

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.axes)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class EncodingAugment(nn.Module):
    def __init__(
        self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1, position_encoder=25
    ):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)
    
class BendrEncoder(nn.Module):
    """
    BENDR convolutional encoder module.

    Args:
        in_features: number of input features / channels
        encoder_h: number of output features in each convolutional operator e.g. number of output features
        enc_width: list of integers indicating the kernel widths for the encoder
        dropout: probability for dropout layer
        projection_head: if projection head should be added such that the number output features should be projected
            to be the same as the number of input features
        enc_downsample: list of integers indicating the strides for the encoder
        grad_frac: float to multiply onto all gradients

    Example:
        >>> from model import BendrEncoder
        >>> import torch
        >>> encoder = BendrEncoder()
        >>> signal = torch.randn(10, 20, 1280)  # batch_size x in_features x signal length
        >>> out = encoder(signal)
        >>> print(out.shape)
        torch.Size([10, 1536, 4])
    """
    def __init__(
        self,
        in_features: int = 20,
        encoder_h: int = 512,
        enc_width: Tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
        enc_downsample: Tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        grad_frac: float = 1.0,
        mask_p_t: float = 0.01,
        mask_p_c: float = 0.005,
        mask_t_span: float = 0.05,
        mask_c_span: float = 0.1,
        encoded_samples: int = 4 * 256
        
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h
        
        ############################################################
        # Primary Encoder
        ############################################################
        
        # Center convolutions
        enc_width = [e if e % 2 else e+1 for e in enc_width]        
        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample, strict=True)):
            self.encoder.add_module(
                f"Encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                    nn.Dropout1d(dropout),  # changed from 2d to 1d compared to bendr
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                )
            )
            in_features = encoder_h

        if grad_frac < 1.0:
            self.register_backward_hook(
                lambda module, in_grad, out_grad: tuple(grad_frac * ig for ig in in_grad)
            )
        
        ############################################################    
        # Encoding Augment
        ############################################################
        
        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span

        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        
        self.enc_augment = EncodingAugment(
            encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span, mask_t_span=mask_t_span
        )
        
        ############################################################
        # Summarizer
        ############################################################
        self.summarizer = nn.AdaptiveAvgPool1d(4)
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return x