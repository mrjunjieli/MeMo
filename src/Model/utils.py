import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import pdb
from torch.nn import MultiheadAttention


EPS = 1e-8


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_parameter(
            "inv_freq", nn.Parameter(inv_freq, requires_grad=False)
        )
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if (
            self.cached_penc is not None
            and self.cached_penc.shape == tensor.shape
        ):
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(
            tensor.shape[0], 1, 1, 1
        )
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


def overlap_and_add(signal, frame_step):
    """Taken from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py
    Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    Example
    -------
    >>> signal = torch.randn(5, 20)
    >>> overlapped = overlap_and_add(signal, 20)
    >>> overlapped.shape
    torch.Size([100])
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(
        frame_length, frame_step
    )  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )

    # frame_old = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.clone().detach().to(signal.device.type)
    # print((frame - frame_old).sum())
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(
        *outer_dimensions, output_subframes, subframe_length
    )
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class AttentionFusion(nn.Module):
    def __init__(self, channel_size):
        super(AttentionFusion, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            channel_size * 2, 1, batch_first=True
        )
        self.visual_weight_proj = nn.Linear(channel_size, 1)
        self.enroll_weight_proj = nn.Linear(channel_size, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enroll_w=None, visual=None):
        # B,L,256
        v_norm = torch.norm(visual, dim=2, p=2, keepdim=True) + EPS
        visual = visual / v_norm
        norm = torch.norm(enroll_w, dim=2, p=2, keepdim=True) + EPS
        enroll_w = enroll_w / norm
        x = torch.cat((visual, enroll_w), 2)
        x, x_weight = self.self_attn(x, x, x)
        visual_score = x[:, :, : visual.shape[2]]
        enroll_score = x[:, :, visual.shape[2] :]

        visual_score = self.visual_weight_proj(visual_score)
        enroll_score = self.enroll_weight_proj(enroll_score)
        att_weight = self.softmax(torch.cat((visual_score, enroll_score), -1))

        fused_emb = (
            att_weight[:, :, 0].unsqueeze(-1) * visual
            + att_weight[:, :, 1].unsqueeze(-1) * enroll_w
        )
        fused_emb = fused_emb.transpose(1, 2)  # B,256,L
        # print(att_weight.shape) # B,L,2
        return fused_emb, att_weight




class Spk_attn(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(Spk_attn, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

    def forward(self, tar,src):
        src = src.transpose(1, 2) # B,C,T -> B, T, C
        tar = tar.transpose(1, 2) # B,C,T -> B, T, C

        src2 = self.self_attn(tar, src, src)[0]

        src2 = src2.transpose(1, 2)  # B, T, C -> B,C,T
        return src2
    

class Select_attn(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(Select_attn, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.linear3 = nn.Linear(d_model, 2)
        

    def forward(self, tar,src):
        src = src.transpose(1, 2) # B,C,T -> B, T, C
        tar = tar.transpose(1, 2) # B,C,T -> B, T, C

        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        class_f = self.linear3(src)

        class_f = class_f.transpose(1, 2)  # B, T, 2 -> B,2,T

        return class_f


if __name__ == "__main__":
    a = torch.randn(10, 100, 256)
    v = torch.randn(10, 100, 256)
    att = AttentionFusion(256)
    att(v, a)
