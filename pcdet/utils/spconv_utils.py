from typing import Optional, Set

import spconv
from spconv.core import ConvAlgo
import sptr
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule

import torch
import torch.nn as nn
import numpy as np


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SubMConv3d(spconv.SubMConv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo: ConvAlgo = None, fp32_accum: bool = None, large_kernel_fast_algo: bool = False, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias, indice_key, algo, fp32_accum, large_kernel_fast_algo, name)
        self._groups = groups
        assert self.out_channels % self._groups == 0
        if self._groups != 1:
            assert in_channels == out_channels
        self.group_stride = self.out_channels // self._groups
        self.in_channels = self.group_stride
        self.out_channels = self.group_stride

    def forward(self, input: spconv.SparseConvTensor, add_input: Optional[spconv.SparseConvTensor] = None):

        grouped_outputs = []
        group_stride = self.group_stride
        for group_idx in range(self._groups):
            cur_feat = input.features[:, group_stride * group_idx:group_stride * (group_idx + 1)]
            group_input = spconv.SparseConvTensor(
                features=cur_feat,
                indices=input.indices,
                spatial_shape=input.spatial_shape,
                batch_size=input.batch_size
            )
            weight = self.weight[group_stride * group_idx:group_stride * (group_idx + 1), ..., group_stride * group_idx:group_stride * (group_idx + 1)].contiguous()
            if self.bias is not None:
                bias = self.bias[group_stride * group_idx:group_stride * (group_idx + 1)]
            else:
                bias = None
            # assert False, (group_input.features.shape, weight.shape, self.weight.shape, input.features.shape)
            group_output = self._conv_forward(
                self.training, group_input,
                weight,
                bias, add_input, name=self.name, sparse_unique_name=self._sparse_unique_name,
                act_type=self.act_type, act_alpha=self.act_alpha, act_beta=self.act_beta
            )
            grouped_outputs.append(group_output)

        return spconv.SparseConvTensor(
            features=torch.cat([go.features for go in grouped_outputs], dim=1),
            indices=grouped_outputs[0].indices,
            spatial_shape=grouped_outputs[0].spatial_shape,
            batch_size=grouped_outputs[0].batch_size
        )


class SparseConv3d(spconv.SparseConv3d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo: ConvAlgo = None, fp32_accum: bool = None, large_kernel_fast_algo: bool = False, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias, indice_key, algo, fp32_accum, large_kernel_fast_algo, name)
        self._groups = groups
        assert self.out_channels % self._groups == 0
        if self._groups != 1:
            assert self.in_channels == self.out_channels
        self.group_stride = self.out_channels // self._groups
        self.in_channels = self.group_stride
        self.out_channels = self.group_stride

    def forward(self, input: spconv.SparseConvTensor, add_input: Optional[spconv.SparseConvTensor] = None):
        grouped_outputs = []
        group_stride = self.group_stride
        for group_idx in range(self._groups):
            cur_feat = input.features[:, group_stride * group_idx:group_stride * (group_idx + 1)]
            group_input = spconv.SparseConvTensor(
                features=cur_feat,
                indices=input.indices,
                spatial_shape=input.spatial_shape,
                batch_size=input.batch_size
            )
            weight = self.weight[group_stride * group_idx:group_stride * (group_idx + 1), ..., group_stride * group_idx:group_stride * (group_idx + 1)].contiguous()
            if self.bias is not None:
                bias = self.bias[group_stride * group_idx:group_stride * (group_idx + 1)]
            else:
                bias = None
            # assert False, (group_input.features.shape, weight.shape, self.weight.shape, input.features.shape)
            group_output = self._conv_forward(
                self.training, group_input,
                weight,
                bias, add_input, name=self.name, sparse_unique_name=self._sparse_unique_name,
                act_type=self.act_type, act_alpha=self.act_alpha, act_beta=self.act_beta
            )
            grouped_outputs.append(group_output)

        return spconv.SparseConvTensor(
            features=torch.cat([go.features for go in grouped_outputs], dim=1),
            indices=grouped_outputs[0].indices,
            spatial_shape=grouped_outputs[0].spatial_shape,
            batch_size=grouped_outputs[0].batch_size
        )
