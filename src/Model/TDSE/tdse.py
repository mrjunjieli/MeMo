import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
sys.path.append('../')
from visual_frontend.visual_frontend import VisualFrontend
from utils import overlap_and_add
import numpy as np
from typing import Tuple
import pdb

EPS = 1e-8


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint
    model_dict = model.state_dict()

    # 1. 检查是否有 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1.1 如果当前模型是多卡（有 'module.' 前缀）但加载的参数没有 'module.' 前缀
        if k.startswith("module.") and not any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = k[len("module.") :]  # 去掉 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.2 如果当前模型是单卡（没有 'module.' 前缀）但加载的参数有 'module.' 前缀
        elif not k.startswith("module.") and any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = "module." + k  # 添加 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.3 当前模型和加载的参数前缀一致
        else:
            new_state_dict[k] = v

    # 2. 检查模型结构是否一致
    for k, v in model_dict.items():
        if k in new_state_dict:
            try:
                model_dict[k].copy_(new_state_dict[k])
            except Exception as e:
                print(f"Error in copying parameter {k}: {e}")
        else:
            print(f"Parameter {k} not found in checkpoint. Skipping...")

    # 3. 更新模型参数
    model.load_state_dict(model_dict)

    return model


def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class av_convtasnet(nn.Module):
    def __init__(
        self,
        N=256,
        L=40,
        B=256,
        H=512,
        P=3,
        X=8,
        R=4,
        C=2,
        sr=16000,
        win=512,
        n_mels=80,
        hop_length=128,
    ):
        super(av_convtasnet, self).__init__()

        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C)
        self.decoder = Decoder(N, L)

        self.visual_frontend = VisualFrontend()
        self.visual_frontend = load_model(
            self.visual_frontend,
            "../Model/visual_frontend/visual_frontend.pt",
        )
        for key, param in self.visual_frontend.named_parameters():
            param.requires_grad = False


    def forward(self, mixture, visual,self_enroll=None,pre_enroll=None):
        batch = visual.shape[0]

        visual = visual.transpose(0, 1)
        visual = visual.unsqueeze(2)
        with torch.no_grad():
            visual = self.visual_frontend(visual)
        visual = visual.transpose(0, 1)

        mixture_w = self.encoder(mixture)

        est_mask = self.separator(mixture_w, visual)

        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        return est_source, None,None 


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(
            1, N, kernel_size=L, stride=L // 2, bias=False
        )

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1)  # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C):
        super(TemporalConvNet, self).__init__()
        self.C = C
        self.layer_norm = ChannelWiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        # Audio TCN
        tcn_blocks = []
        tcn_blocks += [nn.Conv1d(B * 2, B, 1, bias=False)]
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            tcn_blocks += [
                TemporalBlock(
                    B, H, P, stride=1, padding=padding, dilation=dilation
                )
            ]
        self.tcn = _clones(nn.Sequential(*tcn_blocks), R)

        # visual blocks
        ve_blocks = []
        for x in range(5):
            ve_blocks += [VisualConv1D()]
        self.visual_conv = nn.Sequential(*ve_blocks)

        # Audio and visual seprated layers before concatenation
        self.ve_conv1x1 = _clones(nn.Conv1d(512, B, 1, bias=False), R)

        # Mask generation layer
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)
        self.gamma_list = [nn.Parameter(torch.ones((1), device="cuda"))] * R

    def forward(self, x, visual):
        visual = visual.transpose(1, 2)
        visual = self.visual_conv(visual)

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        mixture = x

        batch, Channel, K = x.size()

        for i in range(len(self.tcn)):
            v = self.ve_conv1x1[i](visual)
            v = F.interpolate(v, (32 * v.size()[-1]), mode="linear")
            v = F.pad(v, (0, K - v.size()[-1]))
            v_norm = torch.norm(v, dim=1, p=2, keepdim=True) + EPS
            v_normed = v / v_norm * self.gamma_list[i]

            ref_info = v_normed  # B,256,L
            x = torch.cat((x, ref_info), 1)
            x = self.tcn[i](x)

        x = self.mask_conv1x1(x)
        x = F.relu(x)
        return x


class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(
            512, 512, 3, stride=1, padding=1, dilation=1, groups=512, bias=False
        )
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(
                "{} accept 3D tensor as input".format(self.__name__)
            )
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True
        )  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2))
            .mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation
    ):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(
            out_channels, in_channels, kernel_size, stride, padding, dilation
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation
    ):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = av_convtasnet().cuda()
    mixture = torch.randn(8, 16000).cuda()
    visual = torch.randn(8, 25, 112, 112).cuda()
    model.eval()
    est_source = model(mixture, visual)
    print(est_source[0].shape) # 8,16000


    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 16000).cuda()
    # v_np = torch.randn(1,25,112,112).cuda()
    # flops, params = profile(model, inputs=(x_np, v_np))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))

