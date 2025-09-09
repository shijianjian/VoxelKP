from functools import partial
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from ...utils.spconv_utils import replace_feature, spconv, SparseModule, SubMConv3d, SparseConv3d

import sptr
from sptr.utils import to_3d_numpy, get_indices_params
from sptr.modules import sparse_self_attention, SparseTrTensor
import numpy as np


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if isinstance(self.bn1, (SparseModule,)):
            out = replace_feature(out, self.bn1(out).features)
        else:
            out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        if isinstance(self.bn1, (SparseModule,)):
            out = replace_feature(out, self.bn2(out).features)
        else:
            out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseDSConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseDSConv, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.depth_conv = SubMConv3d(
            inplanes, inplanes, kernel_size=3, groups=8, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.Hardswish()
        self.point_conv = spconv.SubMConv3d(
            inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.depth_conv(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.point_conv(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseMBConv(spconv.SparseModule):
    expansion = 1

    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, norm_fn=None, downsample=None, indice_key=None, mid_channels=None,
        depth_conv_type="subm"
    ):
        super(SparseMBConv, self).__init__()
        mid_channels = mid_channels or round(inplanes * self.expansion)
        if stride != 1:
            last_indice_key = indice_key + "_point"
        else:
            last_indice_key = indice_key

        assert norm_fn is not None
        bias = norm_fn is not None
        self.inverted_conv = spconv.SubMConv3d(
            inplanes, mid_channels, kernel_size=1, stride=1, padding=0, bias=bias, indice_key=indice_key
        )
        if depth_conv_type == "subm":
            self.depth_conv = SubMConv3d(
                mid_channels, mid_channels, kernel_size=kernel_size, groups=8, stride=stride, padding=1, bias=bias, indice_key=indice_key
            )
        elif depth_conv_type == "spconv":
            self.depth_conv = SparseConv3d(
                mid_channels, mid_channels, kernel_size=kernel_size, groups=8, stride=stride, padding=1, bias=bias, indice_key=indice_key)
        else:
            raise RuntimeError()
        self.bn1 = norm_fn(mid_channels)
        self.relu = nn.Hardswish()
        self.point_conv = spconv.SubMConv3d(
            mid_channels, planes, kernel_size=1, stride=1, padding=0, bias=bias, indice_key=last_indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x

        out = self.inverted_conv(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.depth_conv(out)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.point_conv(out)
        out = replace_feature(out, self.bn2(out.features))

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseFusedMBConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, mid_channels=None):
        super(SparseFusedMBConv, self).__init__()
        mid_channels = mid_channels or round(inplanes * self.expansion)

        assert norm_fn is not None
        bias = norm_fn is not None
        self.spatial_conv = spconv.SubMConv3d(
            inplanes, mid_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.Hardswish()
        self.point_conv = spconv.SubMConv3d(
            mid_channels, planes, kernel_size=1, stride=stride, padding=0, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.spatial_conv(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.point_conv(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class LiteMLA(spconv.SparseModule):
    r"""Lightweight multi-scale linear attention."""

    def __init__(
        self, inplanes, planes, qkv_dim=32, heads=4, scales=(5,), indice_key=None
    ):
        super(LiteMLA, self).__init__()

        total_dim = qkv_dim * heads
        self.dim = qkv_dim

        self.qkv = spconv.SubMConv3d(inplanes, 3 * total_dim, 1, bias=False, indice_key=indice_key + "_qkv")
        self.aggreg = nn.ModuleList(
            [
                spconv.SparseSequential(
                    SubMConv3d(
                        3 * total_dim, 3 * total_dim, scale, padding=scale // 2, groups=3 * 8, bias=False
                    ),
                    SubMConv3d(
                        3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=False
                    ),
                )
                for i, scale in enumerate(scales)
            ]
        )
        self.kernel_func = nn.ReLU()

        self.proj = spconv.SubMConv3d(total_dim * (1 + len(scales)), planes, 1, bias=False)
        self.norm = nn.BatchNorm1d(planes)
        self.eps = 1e-15

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = list(qkv.size())

        qkv = qkv.reshape((B, -1, 3 * self.dim, D * H * W))
        qkv = qkv.transpose(-1, -2)
        q, k, v = qkv[..., 0 : self.dim], qkv[..., self.dim : 2 * self.dim], qkv[..., 2 * self.dim :]

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)

        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, D, H, W))
        return out

    # def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    #     # generate multi-scale q, k, v
    #     fake_submanifold_field = x.dense().bool()
    #     assert len(self.aggreg) == 1
    #     qkv = self.qkv(x).dense()
    #     multi_scale_qkv = [qkv]

    #     for op in self.aggreg:
    #         multi_scale_qkv.append(op(qkv))
    #     multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

    #     out = self.relu_linear_att(multi_scale_qkv)
    #     out = out.permute(0, 2, 3, 4, 1)  # To channel last for sparse ops
    #     out = spconv.SparseConvTensor.from_dense(out)
    #     out = self.proj(out)
    #     out = replace_feature(out, self.norm(out.features))
    #     out_dense = out.dense() * fake_submanifold_field
    #     out = spconv.SparseConvTensor.from_dense(out_dense.permute(0, 2, 3, 4, 1))
    #     return out

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        # generate multi-scale q, k, v
        # fake_submanifold_field = x.dense().bool()
        assert len(self.aggreg) == 1
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]

        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))

        multi_scale_qkv = spconv.SparseConvTensor(
            features=torch.cat([o.features for o in multi_scale_qkv], dim=1),
            indices=x.indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        out = self.relu_linear_att(multi_scale_qkv.dense())
        out = out.permute(0, 2, 3, 4, 1)  # To channel last for sparse ops
        out = spconv.SparseConvTensor.from_dense(out)
        out = self.proj(out)
        out = replace_feature(out, self.norm(out.features))
        return out


class EfficientViTBlock(spconv.SparseModule):
    def __init__(self, inplanes: int, dim=32, indice_key=None):
        super(EfficientViTBlock, self).__init__()
        self.context_module = LiteMLA(inplanes, inplanes, qkv_dim=dim, indice_key=indice_key)
        self.local_module = SparseMBConv(inplanes, inplanes, norm_fn=nn.BatchNorm1d, indice_key=indice_key + "_mb")

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x
        x = self.context_module(x)
        x = replace_feature(x, x.features + identity.features)
        x = self.local_module(x)
        return x



class SparseLayerNorm(spconv.SparseModule):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        # self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out_feature = torch.zeros_like(input.features)
        indices_idx = torch.arange(input.indices.size(0), device=input.features.device)
        for batch_idx in range(input.batch_size):
            # Make it (1, seq_len, d_model) then (seq_len, d_model)
            normed_feature = self.ln(input.features[input.indices[:, 0] == batch_idx][None])[0]
            out_feature.index_add_(0, indices_idx[input.indices[:, 0] == batch_idx], normed_feature)
        return input.replace_feature(out_feature)


class SparseAttentionWrapper(spconv.SparseModule):

    def __init__(self, channels, num_heads, indice_key: str, window_size: int = 6, dropout=0., shift_win=False, attn_type="box"):
        super().__init__()

        if isinstance(window_size, (tuple, list,)):
            window_size = np.array(window_size)
        else:
            window_size = np.array([window_size] * 3)

        if attn_type == "box":
            self.attn = sptr.VarLengthMultiheadSA(
                embed_dim=channels,
                num_heads=num_heads,
                indice_key=indice_key,
                window_size=window_size,
                dropout=dropout,
                shift_win=shift_win,
            )
        elif attn_type == "sphere":
            self.attn = SparseMultiheadSASphereConcat(
                embed_dim=channels,
                num_heads=num_heads,
                indice_key=indice_key,
                window_size=window_size,
                window_size_sphere=[1.5, 1.5, 80], 
                quant_size=window_size / 24, 
                quant_size_sphere=[1.5 / 24, 1.5 / 24, 80 / 24], 
                shift_win=False,
                pe_type='contextual',
                rel_query=True, 
                rel_key=True, 
                rel_value=True,
                qkv_bias=True, 
                qk_scale=None, 
                a=0.05 * 0.25,
            )
        else:
            raise NotImplementedError

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        input_tensor = sptr.SparseTrTensor(input.features, input.indices, spatial_shape=input.spatial_shape, batch_size=input.batch_size)
        output_tensor = self.attn(input_tensor).query_feats
        return input.replace_feature(output_tensor)


class SparseAttentionBlock(spconv.SparseModule):

    def __init__(self, channels, num_heads, indice_key: str, window_size: int = 6, dropout=.0, shift_win=False, attn_type="box"):
        super().__init__()

        # This dropout seems not working well.
        self.attn = SparseAttentionWrapper(
            channels=channels,
            num_heads=num_heads,
            dropout=0.,
            indice_key=indice_key + "_sptr_0",
            window_size=window_size,
            shift_win=shift_win,
            attn_type=attn_type,
        )
        self.norm1 = SparseLayerNorm(channels)
        self.norm2 = SparseLayerNorm(channels)
        # self.linear_layer = nn.Linear(channels, channels, bias=True)
        self.relu = nn.GELU()

        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.norm1(input)
        x = self.attn(x)

        feats = input.features + self.drop_path(x.features)
        # For Sparse LayerNorm
        input = input.replace_feature(feats)
        # Shortcut
        feats = input.features + self.drop_path(self.mlp(self.norm2(input).features))

        return input.replace_feature(feats)


def cart2sphere(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = (torch.atan2(y, x) + np.pi) * 180 / np.pi
    beta = torch.atan2(torch.sqrt(x**2 + y**2), z) * 180 / np.pi
    r = torch.sqrt(x**2 + y**2 + z**2)
    return torch.stack([theta, beta, r], -1)


def exponential_split(xyz, index_0, index_1, relative_position_index, a=0.05*0.25):
    '''
    Mapping functioni from r to idx
    | r         ---> idx    |
    | ...       ---> ...    |
    | [-2a, a)  ---> -2     |
    | [-a, 0)   ---> -1     |
    | [0, a)    ---> 0      |
    | [a, 2a)   ---> 1      |
    | [2a, 4a)  ---> 2      |
    | [4a, 6a)  ---> 3      |
    | [6a, 10a) ---> 4      |
    | [10a, 14a)---> 5      |
    | ...       ---> ...    |
    Starting from 0, the split length will double once used twice.
    '''

    r = xyz[:,2]
    rel_pos = r[index_0.long()] - r[index_1.long()] #[M,3]
    rel_pos_abs = rel_pos.abs()
    flag_float = (rel_pos >= 0).float()
    idx = 2 * torch.floor(torch.log((rel_pos_abs+2*a) / a) / np.log(2)) - 2
    idx = idx + ((3*(2**(idx//2)) - 2)*a <= rel_pos_abs).float()
    idx = idx * (2*flag_float - 1) + (flag_float - 1)
    relative_position_index[:, 2] = idx.long() + 24
    return relative_position_index


# Copied From https://github.com/dvlab-research/SphereFormer/blob/master/model/spherical_transformer.py#L64
class SparseMultiheadSASphereConcat(nn.Module):
    def __init__(self, 
        embed_dim, 
        num_heads, 
        indice_key, 
        window_size, 
        window_size_sphere, 
        shift_win=False, 
        pe_type='none', 
        dropout=0., 
        qk_scale=None, 
        qkv_bias=True, 
        algo='native', 
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.indice_key = indice_key
        self.shift_win = shift_win
        self.pe_type = pe_type
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = to_3d_numpy(window_size)
        self.window_size_sphere = to_3d_numpy(window_size_sphere)

        if pe_type == 'contextual':
            self.rel_query, self.rel_key, self.rel_value = kwargs['rel_query'], kwargs['rel_key'], kwargs['rel_value']

            quant_size = kwargs['quant_size']
            self.quant_size = to_3d_numpy(quant_size)

            quant_size_sphere = kwargs['quant_size_sphere']
            self.quant_size_sphere = to_3d_numpy(quant_size_sphere)

            self.a = kwargs['a']

            quant_grid_length = int((window_size[0] + 1e-4)/ quant_size[0])
            assert int((window_size[0] + 1e-4)/ quant_size[0]) == int((window_size[1] + 1e-4)/ quant_size[1])

            # currently only support rel_query, rel_key and rel_value also equal to True
            assert self.rel_query and self.rel_key and self.rel_value
            num_heads_brc1 = num_heads // 2
            self.num_heads_brc1 = num_heads_brc1

            if self.rel_query:
                self.relative_pos_query_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_query_table, std=.02)
            if self.rel_key:
                self.relative_pos_key_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_key_table, std=.02)
            if self.rel_value:
                self.relative_pos_value_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_value_table, std=.02)
            self.quant_grid_length = quant_grid_length

            quant_grid_length_sphere = int((window_size_sphere[0] + 1e-4) / quant_size_sphere[0])
            assert int((window_size_sphere[0] + 1e-4) / quant_size_sphere[0]) == int((window_size_sphere[1] + 1e-4) / quant_size_sphere[1])
            
            num_heads_brc2 = num_heads - num_heads_brc1
            if self.rel_query:
                self.relative_pos_query_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_query_table_sphere, std=.02)
            if self.rel_key:
                self.relative_pos_key_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_key_table_sphere, std=.02)
            if self.rel_value:
                self.relative_pos_value_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_value_table_sphere, std=.02)
            self.quant_grid_length_sphere = quant_grid_length_sphere

        elif pe_type == 'sine':
            normalize_pos_enc = kwargs.get("normalize_pos_enc", True)
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=embed_dim,
                                                       normalize=normalize_pos_enc)
        elif pe_type == "fourier":
            gauss_scale = kwargs.get("gauss_scale", 1.0)
            normalize_pos_enc = kwargs.get("normalize_pos_enc", True)
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=embed_dim,
                                                       gauss_scale=gauss_scale,
                                                       normalize=normalize_pos_enc)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout, inplace=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout, inplace=True)

    def forward(self, sptr_tensor: sptr.SparseTrTensor):
        query, key, value = sptr_tensor.query_feats, sptr_tensor.key_feats, sptr_tensor.value_feats
        assert key is None and value is None
        xyz = sptr_tensor.query_indices[:, 1:]
        batch = sptr_tensor.query_indices[:, 0]

        assert xyz.shape[1] == 3

        N, C = query.shape
        
        qkv = self.qkv(query).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2] #[N, num_heads, C//num_heads]
        query = query * self.scale

        xyz_sphere = cart2sphere(xyz)
        index_params = sptr_tensor.find_indice_params(self.indice_key)
        if index_params is None:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx = get_indices_params(
                xyz, 
                batch, 
                self.window_size, 
                self.shift_win
            )
            index_0_sphere, index_0_offsets_sphere, n_max_sphere, index_1_sphere, index_1_offsets_sphere, sort_idx_sphere = get_indices_params(
                xyz_sphere, 
                batch, 
                self.window_size_sphere, 
                self.shift_win
            )
            sptr_tensor.indice_dict[self.indice_key] = (
                index_0, 
                index_0_offsets, 
                n_max, 
                index_1, 
                index_1_offsets, 
                sort_idx, 
                self.window_size, 
                self.shift_win, 
                index_0_sphere, 
                index_0_offsets_sphere, 
                n_max_sphere, 
                index_1_sphere, 
                index_1_offsets_sphere, 
                sort_idx_sphere
            )
        else:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx, window_size, shift_win, \
                index_0_sphere, index_0_offsets_sphere, n_max_sphere, index_1_sphere, index_1_offsets_sphere, sort_idx_sphere = index_params
            assert (window_size == self.window_size) and (shift_win == self.shift_win), "window_size and shift_win must be the same for sptr_tensors with the same indice_key: {}".format(self.indice_key)
            assert (window_size_sphere == self.window_size_sphere), "window_size and shift_win must be the same for sptr_tensors with the same indice_key: {}".format(self.indice_key)

        kwargs = {"query": query[:, :self.num_heads_brc1].contiguous().float(),
            "key": key[:, :self.num_heads_brc1].contiguous().float(), 
            "value": value[:, :self.num_heads_brc1].contiguous().float(),
            "xyz": xyz.float(),
            "index_0": index_0.int(),
            "index_0_offsets": index_0_offsets.int(),
            "n_max": n_max,
            "index_1": index_1.int(), 
            "index_1_offsets": index_1_offsets.int(),
            "sort_idx": sort_idx,
            "window_size": self.window_size,
            "shift_win": self.shift_win,
            "pe_type": self.pe_type,
        }
        if self.pe_type == 'contextual':
            kwargs.update({
                "rel_query": self.rel_query,
                "rel_key": self.rel_key,
                "rel_value": self.rel_value,
                "quant_size": self.quant_size,
                "quant_grid_length": self.quant_grid_length,
                "relative_pos_query_table": self.relative_pos_query_table.float(),
                "relative_pos_key_table": self.relative_pos_key_table.float(),
                "relative_pos_value_table": self.relative_pos_value_table.float()
            })
        out1 = sparse_self_attention(**kwargs)

        kwargs = {"query": query[:, self.num_heads_brc1:].contiguous().float(),
            "key": key[:, self.num_heads_brc1:].contiguous().float(), 
            "value": value[:, self.num_heads_brc1:].contiguous().float(),
            "xyz": xyz_sphere.float(),
            "index_0": index_0_sphere.int(),
            "index_0_offsets": index_0_offsets_sphere.int(),
            "n_max": n_max_sphere,
            "index_1": index_1_sphere.int(), 
            "index_1_offsets": index_1_offsets_sphere.int(),
            "sort_idx": sort_idx_sphere,
            "window_size": self.window_size_sphere,
            "shift_win": self.shift_win,
            "pe_type": self.pe_type,
        }
        if self.pe_type == 'contextual':
            kwargs.update({
                "rel_query": self.rel_query,
                "rel_key": self.rel_key,
                "rel_value": self.rel_value,
                "quant_size": self.quant_size_sphere,
                "quant_grid_length": self.quant_grid_length_sphere,
                "relative_pos_query_table": self.relative_pos_query_table_sphere.float(),
                "relative_pos_key_table": self.relative_pos_key_table_sphere.float(),
                "relative_pos_value_table": self.relative_pos_value_table_sphere.float(),
                "split_func": partial(exponential_split, a=self.a),
            })
        out2 = sparse_self_attention(**kwargs)

        x = torch.cat([out1, out2], 1).view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x) #[N, C]

        output_tensor = SparseTrTensor(x, sptr_tensor.query_indices, sptr_tensor.spatial_shape, sptr_tensor.batch_size)

        return output_tensor


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
