import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from visual_frontend.visual_frontend import VisualFrontend
from speaker_encoder.ecapa_tdnn import ECAPA_TDNN_c256,ECAPA_TDNN_GLOB_c512,PreEmphasis
from utils import overlap_and_add
import torchaudio 
import pdb
from typing import Tuple
EPS = 1e-8
from torch.nn import MultiheadAttention


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
        self.linear_fusion = nn.Conv1d(
            N * 2, N, kernel_size=1, stride=1, bias=False
        )
        self.preEmphasis = PreEmphasis()
        self.spk_encoder= torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=win, win_length=win, hop_length=hop_length, \
                                        f_min = 20, window_fn=torch.hamming_window, n_mels=n_mels)
        self.spk_model  = ECAPA_TDNN_c256(feat_dim=n_mels,embed_dim=256,pooling_func='ASTP')
        self.spk_model  = load_model(self.spk_model,"../Model/speaker_encoder/ecapa_tdnn.pt")
        for key,param in self.spk_model.named_parameters():
            param.requires_grad = False
        self.spk_self_att = SelfAtten(N)
        self.time_cross_attn = CrossAttention(d_model=N,nhead=1)
        self.slot_attn = CrossAttentionAVG(N)

    def forward(self, mixture, visual=None, self_enroll=[],pre_enroll=[]):
        with torch.no_grad():
            visual = visual.transpose(0, 1)
            visual = visual.unsqueeze(2)
            visual = self.visual_frontend(visual)
            visual = visual.transpose(0, 1)

        mixture_w = self.encoder(mixture)
        if pre_enroll==[]:
            pre_enroll=self_enroll
        if torch.is_tensor(pre_enroll):
            pre_enroll = [pre_enroll]
        if torch.is_tensor(self_enroll):
            self_enroll = [self_enroll]
        spk_mem_bank = []
        for audio in pre_enroll:
            with torch.no_grad():
                spk_emb_input = self.preEmphasis(audio)
                spk_emb_input = self.spk_encoder(spk_emb_input)
                spk_emb_input = spk_emb_input+EPS
                spk_emb_input = spk_emb_input.log()
                spk_emb_input = spk_emb_input - torch.mean(spk_emb_input, dim=-1, keepdim=True)
                spk_emb_input = spk_emb_input.permute(0, 2, 1)
                spk_emb,_ = self.spk_model(spk_emb_input) #B,256
                spk_mem_bank.append(spk_emb.unsqueeze(1))
        spk_mem_bank_tensor = torch.cat(spk_mem_bank,dim=1) #B,slot,256

        #spk bank 
        T = mixture_w.shape[2]
        # Slot dim cross attn 
        spk_emb,attn_map = self.spk_self_att(spk_mem_bank_tensor)
        spk_emb = torch.repeat_interleave(spk_emb, repeats=T, dim=1) #B,T,C
        spk_attn_map = torch.repeat_interleave(attn_map, repeats=T, dim=1) #B,T,slot

        #context bank 
        self_mem_bank = []
        for audio in self_enroll:
            self_mem_bank.append(self.encoder(audio).unsqueeze(1))
        self_mem_bank = torch.cat(self_mem_bank,dim=1)
        # Time dim cross attn 
        mem_bank_after_timeAtt = torch.zeros_like(self_mem_bank) 
        for i in range(self_mem_bank.shape[1]):
            output,attn_weigths = self.time_cross_attn(mixture_w,self_mem_bank[:,i,:,:])
            mem_bank_after_timeAtt[:,i,:,:] = output
        B,Slot,C,T = mem_bank_after_timeAtt.shape
        # Slot dim cross attn 
        self_attn_input = mem_bank_after_timeAtt.transpose(2,3) # B,Slot,T,C 
        self_attn_input = self_attn_input.transpose(1,2) # B,T,Slot,C 
        self_enroll_w,self_attn_map = self.slot_attn(mixture_w.transpose(1,2),self_attn_input,(self_mem_bank.transpose(2,3)).transpose(1,2))
        self_enroll_w = self_enroll_w.transpose(1,2) # B, C, T 
        mixture_w = self.linear_fusion(torch.cat((mixture_w, self_enroll_w), 1))
        
        est_mask = self.separator(mixture_w, visual,spk_emb.transpose(1,2))
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        return est_source, self_attn_map, spk_attn_map 


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

        self.linear_fusion = nn.Conv1d(
            N +256, N, kernel_size=1, stride=1, bias=False)

    def forward(self, x, visual=None,spk_emb=None):
        visual = visual.transpose(1, 2)
        visual = self.visual_conv(visual)

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        batch, Channel, K = x.size()

        spk_emb_rp = spk_emb #B,256,L
        spk_norm = torch.norm(spk_emb_rp,dim=1,p=2,keepdim=True)+EPS
        
        for i in range(len(self.tcn)):
            v = self.ve_conv1x1[i](visual)
            v = F.interpolate(v, (32 * v.size()[-1]), mode="linear")
            v = F.pad(v, (0, K - v.size()[-1]))
            v_norm = torch.norm(v, dim=1, p=2, keepdim=True) + EPS
            v_normed = v / v_norm # B,256,L
            spk_emb_rp_normed = spk_emb_rp/spk_norm
            ref_info = self.linear_fusion(torch.cat((v_normed,spk_emb_rp_normed),dim=1))

            x = torch.cat((x, ref_info * self.gamma_list[i]), 1)
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

class SelfAtten(nn.Module):
    """
    Applies scaled dot-product self-attention on the input tensor.
    This follows the self-attention mechanism described in the "Attention is All You Need" paper.

    Args:
        hidden_dim (int): dimension of the hidden state vector

    Inputs: query, key, value
        - **query** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as queries.
        - **key** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as keys.
        - **value** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as values.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the input sequence.
    """
    def __init__(self, hidden_dim: int) -> None:
        super(SelfAtten, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(256, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scaling = hidden_dim ** 0.5  # Scale factor for dot-product attention

    def forward(self,src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Linear projections for queries, keys, and values
        query = self.query_proj(src) ## B,slot,C 
        key = self.key_proj(src)
        value = self.value_proj(src)
        # T = key.shape[1]
        # query = query.unsqueeze(1)  #B,1,slot,C
        # value = value.unsqueeze(1) #B,T,slot,C
        # key = key.unsqueeze(2) #B,T,1,C
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling  # # shape: B,slot,slot
        
        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)  #  # shape: (B, Slot,Slot)
        attn = torch.mean(attn,dim=1).unsqueeze(1) # B, 1,slot
        # Compute the context vector as a weighted sum of the values
        context = torch.matmul(attn, value) #shape: (B,1,C)

        return context, attn  # #B,1,C,  B，1, Slot


class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        

    def forward(self, tar,src):
        src = src.transpose(1, 2) # B,C,T -> B, T, C
        tar = tar.transpose(1, 2) # B,C,T -> B, T, C

        src2,att_weights = self.self_attn(tar, src, src, attn_mask=None,key_padding_mask=None) #k,q,v 
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.transpose(1,2) # B,T,C -> B, C, T 

        return src,att_weights

class CrossAttentionAVG(nn.Module):
    """
    Applies scaled dot-product self-attention on the input tensor.
    This follows the self-attention mechanism described in the "Attention is All You Need" paper.

    Args:
        hidden_dim (int): dimension of the hidden state vector

    Inputs: query, key, value
        - **query** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as queries.
        - **key** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as keys.
        - **value** (batch_size, seq_len, hidden_dim): tensor containing the input features, acts as values.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the input sequence.
    """
    def __init__(self, hidden_dim: int) -> None:
        super(CrossAttentionAVG, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scaling = hidden_dim ** 0.5  # Scale factor for dot-product attention

    def forward(self, tar: torch.Tensor,src: torch.Tensor,ori_src:torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Linear projections for queries, keys, and values
        query = self.query_proj(src) # B,T,slot,C 
        key = self.key_proj(tar) # B,T,C
        value = self.value_proj(ori_src)

        key = key.unsqueeze(2) # B,T,1,C

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling  # shape: B,T,slot,1 
        
        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-2)  # shape: (B,T, Slot_dim,1)
        attn = attn.transpose(-2,-1)  # shape: (B,T, 1,Slot_dim)
        # Compute the context vector as a weighted sum of the values
        context = torch.matmul(attn, value).squeeze(2)  # shape: (B,T,C)
        return context, attn.squeeze(2) #B,T,C,  B，T, Slot
    

