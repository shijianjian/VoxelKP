import torch
import torch.nn as nn
import numpy as np

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, keypoint_coder_utils, loss_utils, loss_utils_kp
from .pvrcnn_head import PVRCNNHead


class PVRCNNHeadKP(PVRCNNHead):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels=input_channels, model_cfg=model_cfg, num_class=num_class, **kwargs)

        self.kp_coder = getattr(keypoint_coder_utils, self.model_cfg.TARGET_CONFIG.KEYPOINT_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('KEYPOINT_CODER_CONFIG', {})
        )

        self.kp_reg_layers = self.make_fc_layers(
            input_channels=self.reg_layers[0].in_channels,
            output_channels=self.kp_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

    def get_kp_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.kp_coder.code_size
        box_code_size = self.box_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_kp3d_ct = forward_ret_dict['gt_of_kp'][..., 0:code_size]
        kp_mask = forward_ret_dict["batch_gt_of_kp_mask"].view(-1, code_size // 3)  # (rcnn_batch_size, C)
        rcnn_reg_kp = forward_ret_dict['rcnn_reg_kp']  # (rcnn_batch_size, C)
        rcnn_reg_kp = torch.stack([
            rcnn_reg_kp[...,0::3],
            rcnn_reg_kp[...,1::3],
            rcnn_reg_kp[...,2::3]
        ], dim=-1)  # (rcnn_batch_size, codesize, 3)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = forward_ret_dict['gt_of_rois'][..., 0:box_code_size].view(-1, box_code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}
        if loss_cfgs.REG_LOSS_KP == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, box_code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.kp_coder.encode_torch(
                gt_kp3d_ct.view(rcnn_batch_size, code_size // 3, 3), rois_anchor
            )
            rcnn_kp_loss_reg = self.reg_kp_loss_func(
                rcnn_reg_kp.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.view(rcnn_batch_size, -1).unsqueeze(dim=0),
            )  # [B, M, 42]

            rcnn_kp_loss_reg = (rcnn_kp_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_kp_loss_reg = rcnn_kp_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_kp_reg_weight']
            tb_dict['rcnn_kp_loss_reg'] = rcnn_kp_loss_reg.item()

        elif loss_cfgs.REG_LOSS_KP == 'oks':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, box_code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.kp_coder.encode_torch(
                gt_kp3d_ct.view(rcnn_batch_size, code_size // 3, 3), rois_anchor
            )
            rcnn_kp_loss_reg = self.reg_kp_loss_func(
                rcnn_reg_kp.view(rcnn_batch_size, -1, 3),
                reg_targets.view(rcnn_batch_size, -1, 3),
                mask=kp_mask,
                bboxes=rois_anchor
            )  # [B, M]

            rcnn_kp_loss_reg = (rcnn_kp_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_kp_loss_reg = rcnn_kp_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_kp_reg_weight']
            tb_dict['rcnn_kp_loss_reg'] = rcnn_kp_loss_reg.item()

        else:
            raise NotImplementedError

        return rcnn_kp_loss_reg, tb_dict

    def build_losses(self, losses_cfg):
        super().build_losses(losses_cfg)
        if losses_cfg.REG_LOSS_KP == 'smooth-l1':
            self.add_module(
                'reg_kp_loss_func',
                loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights_kp'])
            )
        elif losses_cfg.REG_LOSS_KP == 'oks':
            self.add_module(
                'reg_kp_loss_func',
                loss_utils_kp.OksLoss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights_kp'])
            )
        else:
            raise NotImplementedError

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict=tb_dict)

        rcnn_kp_loss_reg, reg_tb_dict = self.get_kp_reg_layer_loss(self.forward_ret_dict)

        rcnn_loss += rcnn_kp_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_keypoints(self, batch_size, rois, kp_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            kp_preds: (BN, code_size)
        """
        code_size = self.kp_coder.code_size
        batch_kp_preds = torch.stack([
            kp_preds[..., 0::3],
            kp_preds[..., 1::3],
            kp_preds[..., 2::3],
        ], dim=-1)
        batch_kp_preds = kp_preds.view(batch_size, -1, code_size // 3, 3)

        roi_xyz = rois[:, :, 0:3]
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_kp_preds = self.kp_coder.decode_torch(batch_kp_preds, local_rois)
        batch_kp_preds[..., 0:3] += roi_xyz.unsqueeze(-2)

        # batch_kp_preds = batch_kp_preds.view(batch_size, -1, code_size // 3, 3)
        return batch_kp_preds

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        # assert False, (targets_dict.keys(), targets_dict['gt_of_rois'].shape, targets_dict['gt_of_kp'].shape)
        rois = targets_dict['rois']  # (B, N, 7 + C)
        rois_kp = targets_dict['rois_kp']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        gt_of_kp = targets_dict['gt_of_kp']  # (B, N, 14, 3)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        targets_dict['gt_of_kp_src'] = gt_of_kp.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois

        roi_center_kp = rois_kp[:, :, 0:3]
        # roi_ry_kp = rois_kp[:, :, 6] % (2 * np.pi)
        gt_of_kp[:, :, :, 0:3] = gt_of_kp[:, :, :, 0:3] - roi_center_kp[..., None, :]

        # TODO: check if need to rotate
        # gt_of_kp = common_utils.rotate_points_along_z(
        #     points=gt_of_kp.view(-1, 1, gt_of_kp.shape[-1]), angle=-roi_ry_kp.view(-1)
        # ).view(batch_size, -1, gt_of_kp.shape[-1])
        targets_dict['gt_of_kp'] = gt_of_kp

        return targets_dict

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['rois_kp'] = targets_dict['rois_kp']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        rcnn_reg_kp = self.kp_reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_kp_preds = self.generate_predicted_keypoints(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois_kp'], kp_preds=rcnn_reg_kp
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_kp_preds'] = batch_kp_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rcnn_reg_kp'] = rcnn_reg_kp

            self.forward_ret_dict = targets_dict
        return batch_dict
