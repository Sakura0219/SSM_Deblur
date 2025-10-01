import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np
from einops.layers.torch import Rearrange
from functools import partial
from timm.layers import trunc_normal_tf_
from timm.models import named_apply


# 导入MDFA模块
class tongdao(nn.Module):  # 处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(
            y, size=(x.size(2), x.size(3)), mode="nearest"
        )  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准


class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(
            in_channel, 1, kernel_size=1, bias=False
        )  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励


class hebing(nn.Module):  # 函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(
            U_tongdao, U_kongjian
        )  # 取两者的逐元素最大值，结合通道和空间激励


class MDFA(nn.Module):  ##多尺度空洞融合注意力模块。
    def __init__(
        self, dim_in, dim_out, rate=1, bn_mom=0.1
    ):  # 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = (
            nn.Sequential(  # 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        )
        self.branch2 = (
            nn.Sequential(  # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    3,
                    1,
                    padding=6 * rate,
                    dilation=6 * rate,
                    bias=True,
                ),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        )
        self.branch3 = (
            nn.Sequential(  # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    3,
                    1,
                    padding=12 * rate,
                    dilation=12 * rate,
                    bias=True,
                ),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        )
        self.branch4 = (
            nn.Sequential(  # 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    3,
                    1,
                    padding=18 * rate,
                    dilation=18 * rate,
                    bias=True,
                ),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        )
        self.branch5_conv = nn.Conv2d(
            dim_in, dim_out, 1, 1, 0, bias=True
        )  # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(  # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing = hebing(in_channel=dim_out * 5)  # 整合通道和空间特征的合并模块

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(
            global_feature, (row, col), None, "bilinear", True
        )
        # 合并所有特征
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1
        )
        # 应用合并模块进行通道和空间特征增强
        larry = self.Hebing(feature_cat)
        larry_feature_cat = larry * feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(larry_feature_cat)

        return result


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
            + self.bias
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=1, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c h w -> b h w c")
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, "b h w c -> b c h w")

        return out


##########################################################################
## LKGGF模块：大核分组门控融合模块
class LGAG(nn.Module):
    """大核分组注意力门控模块"""

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation="relu"):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class PagFM(nn.Module):
    """部分特征融合模块"""

    def __init__(
        self,
        in_channels,
        mid_channels,
        after_relu=False,
        with_channel=True,
        BatchNorm=nn.BatchNorm2d,
    ):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q,
            size=[input_size[2], input_size[3]],
            mode="bilinear",
            align_corners=False,
        )
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(
            y, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False
        )
        x = (1 - sim_map) * x + sim_map * y

        return x


class LKGGF(nn.Module):
    """大核分组门控融合模块"""

    def __init__(self, in_channels, kernel_size=7, groups=4):
        super(LKGGF, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.groups = groups

        # 大核深度可分离卷积
        self.large_kernel_dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.large_kernel_pw = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )
        self.large_kernel_bn = nn.BatchNorm2d(in_channels)

        # LGAG模块
        self.lgag = LGAG(
            F_g=in_channels,
            F_l=in_channels,
            F_int=in_channels // 2,
            kernel_size=kernel_size,
            groups=groups,
        )

        # PagFM模块
        self.pagfm = PagFM(in_channels=in_channels, mid_channels=in_channels // 2)

        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 残差连接
        self.residual_conv = (
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            if in_channels != in_channels
            else nn.Identity()
        )

    def forward(self, x, skip=None):
        # 大核卷积分支
        large_feat = self.large_kernel_dw(x)
        large_feat = self.large_kernel_pw(large_feat)
        large_feat = self.large_kernel_bn(large_feat)

        # LGAG分支
        if skip is not None:
            lgag_feat = self.lgag(skip, x)
        else:
            lgag_feat = x

        # PagFM分支
        if skip is not None:
            pagfm_feat = self.pagfm(x, skip)
        else:
            pagfm_feat = x

        # 特征融合
        combined_feat = large_feat + lgag_feat + pagfm_feat

        # 门控控制
        gate_weight = self.gate(combined_feat)
        out = gate_weight * combined_feat + (1 - gate_weight) * x

        # 残差连接
        out = out + self.residual_conv(x)

        return out


class EVS(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor=3,
        bias=False,
        LayerNorm_type="WithBias",
        att=False,
        idx=3,
        patch=128,
    ):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx

        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)

        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.att:
            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- EVSSM -----------------------
class EVSSM(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        ffn_expansion_factor=3,
        bias=False,
        use_lkggf=False,
        use_mdfa=False,  # 添加MDFA使用标志
    ):
        super(EVSSM, self).__init__()

        self.encoder = True
        self.use_lkggf = use_lkggf
        self.use_mdfa = use_mdfa  # 存储MDFA使用标志

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=384,
            )
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(
                dim=dim * 2,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=192,
            )
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(
                dim=dim * 4,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=96,
            )
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(
                dim=dim * 4,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=96,
            )
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2**2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(
                dim=dim * 2,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=192,
            )
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                att=True,
                idx=i,
                patch=384,
            )
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        # LKGGF模块用于跳跃连接
        if self.use_lkggf:
            self.lkggf_level2 = LKGGF(dim * 2)
            self.lkggf_level1 = LKGGF(dim)

        # MDFA模块插入在解码器level2和level1之间
        if self.use_mdfa:
            self.mdfa_decoder3 = MDFA(dim * 4, dim * 4) #192
            self.mdfa_decoder2 = MDFA(dim * 2, dim * 2)  # 96维输入输出
            self.mdfa_decoder1 = MDFA(dim, dim)  # 48维

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(
            inp_img
        )  # inp_img [1,3,384,384] inp_enc_level1 [1,48,384,384]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [1,48,384,384]

        inp_enc_level2 = self.down1_2(out_enc_level1)  # [1,96,192,192]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # [1,192,96,96]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # [1,192,96,96]
        # decode
        out_dec_level3 = self.decoder_level3(out_enc_level3)  # [1,192,96,96]

        # MDFA模块处理解码器level3输出
        if self.use_mdfa:
            out_dec_level3 = self.mdfa_decoder3(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)  # [1,96,192,192]

        # 使用LKGGF模块处理跳跃连接
        if self.use_lkggf:
            inp_dec_level2 = self.lkggf_level2(inp_dec_level2, out_enc_level2)
        else:
            inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [1,dim=96=2*dim,192,192]

        # MDFA模块处理解码器level2输出
        if self.use_mdfa:
            out_dec_level2 = self.mdfa_decoder2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        # 使用LKGGF模块处理跳跃连接
        if self.use_lkggf:
            inp_dec_level1 = self.lkggf_level1(inp_dec_level1, out_enc_level1)
        else:
            inp_dec_level1 = inp_dec_level1 + out_enc_level1

        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # [1,48,384,384]
        # MDFA模块处理解码器level1输出
        if self.use_mdfa:
            out_dec_level1 = self.mdfa_decoder1(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img  # [1,3,384,384]

        return out_dec_level1


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = EVSSM(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2, 2, 2],  # 减少块数量，加快运行速度
        ffn_expansion_factor=3,
        bias=False,
    )
    model.to(device)
    # 设置为评估模式
    model.eval()

    # 创建随机输入张量 (Batch=1, Channels=3, Height=384, Width=384)
    input_tensor = torch.randn(1, 3, 384, 384)
    input_tensor = input_tensor.to(device)
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    print(output.shape)
