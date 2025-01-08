from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import (
    post_act_block, SparseDSConv, SparseMBConv, SparseFusedMBConv, SparseBasicBlock,
    EfficientViTBlock, SparseAttentionBlock
)

    
class SparseGlobalAvgPool(spconv.SparseModule):
    def forward(self, input):
        batch_size = input.batch_size
        ft = []
        for b in range(batch_size):
            ft.append(torch.mean(input.features[[input.indices[:, 0] == b]], dim=0, keepdim=True))
        return torch.cat(ft, dim=0)


class ChannelAttention(spconv.SparseModule):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = SparseGlobalAvgPool()

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # assert False, (x.features.shape, self.avg_pool(x).shape)
        out = self.sigmoid(self.fc(self.avg_pool(input)))
        indices_idx = torch.arange(input.indices.size(0), device=input.features.device)
        out_tensor = torch.zeros_like(input.features)
        for b in range(input.batch_size):
            out_tensor.index_add_(
                0,
                indices_idx[input.indices[:, 0] == b],
                input.features[input.indices[:, 0] == b] * out[b:b + 1]
            )
        return replace_feature(input, out_tensor)


class SparseSEBlock(SparseBasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super().__init__(inplanes, planes, stride, bias, norm_fn, downsample, indice_key)
        self.ca = ChannelAttention(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        out = self.ca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseSEBlock2D(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, indice_key=None):
        super(SparseSEBlock2D, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.ca(out)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseSKBEV(spconv.SparseModule):

    def __init__(self, inplanes, depth, bias=False, norm_fn=None, downsample=None, indice_key=None, ratio=16):
        super(SparseSKBEV, self).__init__()
        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=bias, indice_key=indice_key + "_sk5"
        )
        self.bn2 = norm_fn(inplanes)
        self.downsample = downsample

        self.squeeze = nn.Sequential(
            nn.Linear(inplanes, inplanes // ratio, bias=False),
            nn.BatchNorm1d(inplanes // ratio),
            nn.ReLU(inplace=True)
        )
        self.excitation3 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation5 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.to_bev = spconv.SparseSequential(
            spconv.SparseConv3d(
                inplanes, inplanes, kernel_size=(depth, 3, 3), padding=(0, 1, 1), bias=bias, indice_key=indice_key + "_bev"
            ),
            norm_fn(inplanes),
            nn.ReLU(),
        )

    def forward(self, x):

        out3 = self.conv1(x)
        out3 = replace_feature(out3, self.relu(self.bn1(out3.features)))
        out5 = self.conv2(x)
        out5 = replace_feature(out5, self.relu(self.bn2(out5.features)))

        feat_u = out3.features + out5.features
        batch_size = out5.batch_size
        ft = []
        for b in range(batch_size):
            ft.append(torch.mean(feat_u[[out5.indices[:, 0] == b]], dim=0, keepdim=True))
        feat_s = torch.cat(ft, dim=0)
        feat_z = self.squeeze(feat_s)

        ext3 = self.excitation3(feat_z)
        ext5 = self.excitation5(feat_z)

        attention_vector = self.softmax(torch.stack([ext3, ext5], dim=1))

        indices_idx = torch.arange(out3.indices.size(0), device=out3.features.device)

        out3_tensor = torch.zeros_like(out3.features)
        out5_tensor = torch.zeros_like(out3.features)
        for b in range(out3.batch_size):
            out3_tensor.index_add_(
                0,
                indices_idx[out3.indices[:, 0] == b],
                out3.features[out3.indices[:, 0] == b] * attention_vector[b:b + 1, 0]
            )
            out5_tensor.index_add_(
                0,
                indices_idx[out5.indices[:, 0] == b],
                out5.features[out5.indices[:, 0] == b] * attention_vector[b:b + 1, 1]
            )

        return self.to_bev(replace_feature(out3, out3_tensor + out5_tensor))


class SparseSKConv(spconv.SparseModule):

    expansion = 1

    def __init__(self, inplanes, stride=1, bias=False, norm_fn=None, downsample=None, indice_key=None, ratio=16):
        super(SparseSKConv, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=5, stride=stride, padding=2, bias=bias, indice_key=indice_key + "_sk5"
        )
        self.bn2 = norm_fn(inplanes)
        self.downsample = downsample
        self.stride = stride

        self.squeeze = nn.Sequential(
            nn.Linear(inplanes, inplanes // ratio, bias=False),
            nn.BatchNorm1d(inplanes // ratio),
            nn.ReLU(inplace=True)
        )
        self.excitation3 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation5 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out3 = self.conv1(x)
        out3 = replace_feature(out3, self.relu(self.bn1(out3.features)))
        out5 = self.conv2(x)
        out5 = replace_feature(out5, self.relu(self.bn2(out5.features)))

        feat_u = out3.features + out5.features
        batch_size = out5.batch_size
        ft = []
        for b in range(batch_size):
            ft.append(torch.mean(feat_u[[out5.indices[:, 0] == b]], dim=0, keepdim=True))
        feat_s = torch.cat(ft, dim=0)
        feat_z = self.squeeze(feat_s)

        ext3 = self.excitation3(feat_z)
        ext5 = self.excitation5(feat_z)

        attention_vector = self.softmax(torch.stack([ext3, ext5], dim=1))

        indices_idx = torch.arange(out3.indices.size(0), device=out3.features.device)

        out3_tensor = torch.zeros_like(out3.features)
        out5_tensor = torch.zeros_like(out3.features)
        for b in range(out3.batch_size):
            out3_tensor.index_add_(
                0,
                indices_idx[out3.indices[:, 0] == b],
                out3.features[out3.indices[:, 0] == b] * attention_vector[b:b + 1, 0]
            )
            out5_tensor.index_add_(
                0,
                indices_idx[out5.indices[:, 0] == b],
                out5.features[out5.indices[:, 0] == b] * attention_vector[b:b + 1, 1]
            )

        return replace_feature(out3, out3_tensor + out5_tensor)


class SparseSKBEV3Way(spconv.SparseModule):

    expansion = 1

    def __init__(self, inplanes, depth, stride=1, bias=False, norm_fn=None, downsample=None, indice_key=None, ratio=16):
        super(SparseSKBEV3Way, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(inplanes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=bias, indice_key=indice_key + "_sk3"
        )
        self.bn2 = norm_fn(inplanes)

        self.conv3 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=bias, indice_key=indice_key + "_sk5"
        )
        self.bn3 = norm_fn(inplanes)

        self.downsample = downsample

        self.squeeze = nn.Sequential(
            nn.Linear(inplanes, inplanes // ratio, bias=False),
            nn.BatchNorm1d(inplanes // ratio),
            nn.ReLU(inplace=True)
        )
        self.excitation3 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation5 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation7 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.to_bev = spconv.SparseSequential(
            spconv.SparseConv3d(
                inplanes, inplanes, kernel_size=(depth, 3, 3), padding=(0, 1, 1), bias=bias, indice_key=indice_key + "_bev"
            ),
            norm_fn(inplanes),
            nn.ReLU(),
        )

    def forward(self, x):

        out3 = self.conv1(x)
        out3 = replace_feature(out3, self.relu(self.bn1(out3.features)))
        out5 = self.conv2(x)
        out5 = replace_feature(out5, self.relu(self.bn2(out5.features)))
        out7 = self.conv3(x)
        out7 = replace_feature(out7, self.relu(self.bn3(out7.features)))

        feat_u = out3.features + out5.features + out7.features
        batch_size = out5.batch_size
        ft = []
        for b in range(batch_size):
            ft.append(torch.mean(feat_u[[out5.indices[:, 0] == b]], dim=0, keepdim=True))
        feat_s = torch.cat(ft, dim=0)
        feat_z = self.squeeze(feat_s)

        ext3 = self.excitation3(feat_z)
        ext5 = self.excitation5(feat_z)
        ext7 = self.excitation7(feat_z)

        attention_vector = self.softmax(torch.stack([ext3, ext5, ext7], dim=1))

        indices_idx = torch.arange(out3.indices.size(0), device=out3.features.device)

        out3_tensor = torch.zeros_like(out3.features)
        out5_tensor = torch.zeros_like(out3.features)
        out7_tensor = torch.zeros_like(out3.features)
        for b in range(out3.batch_size):
            out3_tensor.index_add_(
                0,
                indices_idx[out3.indices[:, 0] == b],
                out3.features[out3.indices[:, 0] == b] * attention_vector[b:b + 1, 0]
            )
            out5_tensor.index_add_(
                0,
                indices_idx[out5.indices[:, 0] == b],
                out5.features[out5.indices[:, 0] == b] * attention_vector[b:b + 1, 1]
            )
            out7_tensor.index_add_(
                0,
                indices_idx[out7.indices[:, 0] == b],
                out7.features[out7.indices[:, 0] == b] * attention_vector[b:b + 1, 2]
            )

        return self.to_bev(replace_feature(out3, out3_tensor + out5_tensor + out7_tensor))


class SparseSKConv3Way(spconv.SparseModule):

    expansion = 1

    def __init__(self, inplanes, stride=1, bias=False, norm_fn=None, downsample=None, indice_key=None, ratio=16):
        super(SparseSKConv3Way, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(inplanes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=5, stride=stride, padding=2, bias=bias, indice_key=indice_key + "_sk5"
        )
        self.bn2 = norm_fn(inplanes)

        self.conv3 = spconv.SubMConv3d(
            inplanes, inplanes, kernel_size=7, stride=stride, padding=3, bias=bias, indice_key=indice_key + "_sk7"
        )
        self.bn3 = norm_fn(inplanes)

        self.downsample = downsample
        self.stride = stride

        self.squeeze = nn.Sequential(
            nn.Linear(inplanes, inplanes // ratio, bias=False),
            nn.BatchNorm1d(inplanes // ratio),
            nn.ReLU(inplace=True)
        )
        self.excitation3 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation5 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.excitation7 = nn.Linear(inplanes // ratio, inplanes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out3 = self.conv1(x)
        out3 = replace_feature(out3, self.relu(self.bn1(out3.features)))
        out5 = self.conv2(x)
        out5 = replace_feature(out5, self.relu(self.bn2(out5.features)))
        out7 = self.conv3(x)
        out7 = replace_feature(out7, self.relu(self.bn3(out7.features)))

        feat_u = out3.features + out5.features + out7.features
        batch_size = out5.batch_size
        ft = []
        for b in range(batch_size):
            ft.append(torch.mean(feat_u[[out5.indices[:, 0] == b]], dim=0, keepdim=True))
        feat_s = torch.cat(ft, dim=0)
        feat_z = self.squeeze(feat_s)

        ext3 = self.excitation3(feat_z)
        ext5 = self.excitation5(feat_z)
        ext7 = self.excitation7(feat_z)

        attention_vector = self.softmax(torch.stack([ext3, ext5, ext7], dim=1))

        indices_idx = torch.arange(out3.indices.size(0), device=out3.features.device)

        out3_tensor = torch.zeros_like(out3.features)
        out5_tensor = torch.zeros_like(out3.features)
        out7_tensor = torch.zeros_like(out3.features)
        for b in range(out3.batch_size):
            out3_tensor.index_add_(
                0,
                indices_idx[out3.indices[:, 0] == b],
                out3.features[out3.indices[:, 0] == b] * attention_vector[b:b + 1, 0]
            )
            out5_tensor.index_add_(
                0,
                indices_idx[out5.indices[:, 0] == b],
                out5.features[out5.indices[:, 0] == b] * attention_vector[b:b + 1, 1]
            )
            out7_tensor.index_add_(
                0,
                indices_idx[out7.indices[:, 0] == b],
                out7.features[out7.indices[:, 0] == b] * attention_vector[b:b + 1, 2]
            )

        return replace_feature(out3, out3_tensor + out5_tensor + out7_tensor)


class SparseSKBlock(SparseBasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None, three_way=False):
        super().__init__(inplanes, planes, stride, bias, norm_fn, downsample, indice_key)
        if three_way:
            self.sk = SparseSKConv3Way(inplanes, norm_fn=norm_fn, indice_key=indice_key, ratio=8)
        else:
            self.sk = SparseSKConv(inplanes, norm_fn=norm_fn, indice_key=indice_key, ratio=8)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.sk(out)

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class MLPResidual(spconv.SparseModule):

    def __init__(self, voxel_conv, planes):
        super().__init__()
        self.point_conv = spconv.SparseSequential(
            nn.Linear(planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )
        self.voxel_conv = voxel_conv

    def forward(self, x):
        point_x = self.point_conv(x)
        voxel_x = self.voxel_conv(x)
        return point_x + voxel_x


class VoxelResBackBone8xVoxelNeXtEffv2Next4(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)
        bev_depth = model_cfg.get('BEV_DEPTH', [8, 4, 2])
        three_way = model_cfg.get('THREE_WAY_SKBLOCK', False)
        num_heads = model_cfg.get('NUM_ATTENTION_HEADS', 4)
        window_size = model_cfg.get('ATTENTION_WINDOW_SIZE', [6, 6])
        block = post_act_block

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            # SparseMBConv(channels[0], channels[1], norm_fn=norm_fn, kernel_size=spconv_kernel_sizes[0], stride=2, depth_conv_type="spconv", indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            # SparseMBConv(channels[1], channels[2], norm_fn=norm_fn, kernel_size=spconv_kernel_sizes[1], stride=2, depth_conv_type="spconv", indice_key='res3_1'),
            MLPResidual(
                spconv.SparseSequential(
                    SparseSKBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3', three_way=False),
                    SparseSKBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3', three_way=False),
                ),
                channels[2]
            )
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            MLPResidual(
                spconv.SparseSequential(
                    SparseSKBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4', three_way=False),
                    SparseSKBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4', three_way=False),
                ),
                channels[3]
            )
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            # SparseMBConv(channels[4], channels[4], norm_fn=norm_fn, kernel_size=spconv_kernel_sizes[3], stride=2, depth_conv_type="spconv", indice_key='res5_1'),
            MLPResidual(
                spconv.SparseSequential(
                    SparseSKBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5', three_way=False),
                    SparseAttentionBlock(channels[4], num_heads=num_heads, indice_key='res5_att', window_size=window_size[0], dropout=0.),
                    SparseSKBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5', three_way=False),
                    SparseAttentionBlock(channels[4], num_heads=num_heads, indice_key='res5_att', window_size=window_size[0], dropout=0.),
                ),
                channels[4]
            )
        )

        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            # SparseMBConv(channels[4], channels[4], norm_fn=norm_fn, kernel_size=spconv_kernel_sizes[3], stride=2, depth_conv_type="spconv", indice_key='res6_1'),
            MLPResidual(
                spconv.SparseSequential(
                    SparseSKBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6', three_way=False),
                    SparseAttentionBlock(channels[4], num_heads=num_heads, indice_key='res6_att', window_size=window_size[1], dropout=0.),
                    SparseSKBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6', three_way=False),
                    SparseAttentionBlock(channels[4], num_heads=num_heads, indice_key='res6_att', window_size=window_size[1], dropout=0.),
                ),
                channels[4]
            )
        )

        self.conv4_bev = spconv.SparseSequential(
            spconv.SparseConv3d(channels[3], out_channel, kernel_size=(bev_depth[0], 1, 1), bias=False, indice_key="conv4_bev"),
            norm_fn(out_channel),
            nn.ReLU(),
        )
        self.conv5_bev = spconv.SparseSequential(
            spconv.SparseConv3d(channels[4], out_channel, kernel_size=(bev_depth[1], 1, 1), bias=False, indice_key="conv5_bev"),
            norm_fn(out_channel),
            nn.ReLU(),
        )
        self.conv6_bev = spconv.SparseSequential(
            spconv.SparseConv3d(channels[4], out_channel, kernel_size=(bev_depth[2], 1, 1), bias=False, indice_key="conv6_bev"),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down_xy2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

    def bev_out(self, x_conv, axis="z"):
        # voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        features_cat = x_conv.features
        feature_num_count = features_cat.shape[1]
        if axis == "x":
            axis_indices = [0, 1, 2]
            spatial_shape = x_conv.spatial_shape[:-1]
        elif axis == "y":
            axis_indices = [0, 1, 3]
            spatial_shape = x_conv.spatial_shape[:1] + x_conv.spatial_shape[2:]
        elif axis == "z":
            features_cat = features_cat * ((x_conv.indices[:, 1:2] + 1) / x_conv.spatial_shape[0])
            axis_indices = [0, 2, 3]
            spatial_shape = x_conv.spatial_shape[1:]
        else:
            raise RuntimeError
        indices_cat = x_conv.indices[:, axis_indices]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], feature_num_count))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def bev_conv(self, feat_1, feat_2, feat_3):
        bev_4 = self.conv4_bev(feat_1)
        bev_5 = self.conv5_bev(feat_2)
        bev_6 = self.conv6_bev(feat_3)
        assert bev_4.spatial_shape[0] == bev_5.spatial_shape[0] == bev_6.spatial_shape[0] == 1, (bev_4.spatial_shape[0], bev_5.spatial_shape[0], bev_6.spatial_shape[0])

        bev_5.indices[:, 1:] *= 2
        bev_6.indices[:, 1:] *= 4
        bev_5.indices[:, 1:] += 1  # move x,y,z offset
        bev_6.indices[:, 1:] += 2  # move x,y,z offset
        bev_4 = bev_4.replace_feature(torch.cat([bev_4.features, bev_5.features, bev_6.features]))
        bev_4.indices = torch.cat([bev_4.indices, bev_5.indices, bev_6.indices])

        return self.bev_out(bev_4, axis="z")

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
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)
        
        out = self.bev_conv(x_conv4, x_conv5, x_conv6)
        # x_conv5.indices[:, 1:] *= 2
        # x_conv6.indices[:, 1:] *= 4
        # x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        # x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        # assert False, (voxel_features.shape, voxel_coords.shape, self.sparse_shape, x_conv1.spatial_shape, x_conv2.spatial_shape, x_conv3.spatial_shape, x_conv4.spatial_shape, len(torch.unique(x_conv4.indices[:, 1])), len(torch.unique(x_conv4.indices[:, 2])), len(torch.unique(x_conv4.indices[:, 3])))

        # out = self.bev_out(x_conv4, "z")
        # bev_yz = self.bev_out(x_conv4, "x")
        # bev_xz = self.bev_out(x_conv4, "y")
        # assert False, indices_align(out, bev_yz, "y")
        
        # out = self.cross_attention(out, bev_yz, bev_xz)
        # assert False, (out.features.shape, out.indices.shape, out.spatial_shape, out.batch_size)
        out = self.conv_out(out)
        out = self.shared_conv(out)

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
