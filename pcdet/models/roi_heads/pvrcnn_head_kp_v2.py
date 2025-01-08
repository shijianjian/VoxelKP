import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, keypoint_coder_utils, loss_utils
from .pvrcnn_head_kp import PVRCNNHeadKP

# KP estimation came along with the boxes
class PVRCNNHeadKPV2(PVRCNNHeadKP):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels=input_channels, model_cfg=model_cfg, num_class=num_class, **kwargs)

        self.kp_coder = getattr(keypoint_coder_utils, self.model_cfg.TARGET_CONFIG.KEYPOINT_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('KEYPOINT_CODER_CONFIG', {})
        )

        # self.kp_reg_layers = self.make_fc_layers(
        #     input_channels=self.reg_layers[0].in_channels,
        #     output_channels=self.kp_coder.code_size * self.num_class,
        #     fc_list=self.model_cfg.REG_FC
        # )
        self.reg_layers = self.make_fc_layers(
            input_channels=self.cls_layers[0].in_channels,
            output_channels=(self.box_coder.code_size + self.kp_coder.code_size) * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

    def build_losses(self, losses_cfg):
        super().build_losses(losses_cfg)
        self.add_module(
            'reg_kp_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights_kp'])
        )

    def get_loss(self, tb_dict=None):
        return super().get_loss(tb_dict=tb_dict)

    def generate_predicted_keypoints(self, batch_size, rois, kp_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            kp_preds: (BN, code_size)
        """
        code_size = self.kp_coder.code_size
        batch_kp_preds = kp_preds.view(batch_size, -1, code_size // 3, 3)

        roi_xyz = rois[:, :, 0:3]
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_kp_preds = self.kp_coder.decode_torch(batch_kp_preds, local_rois)
        batch_kp_preds[..., 0:3] += roi_xyz.unsqueeze(-2)

        batch_kp_preds = batch_kp_preds.view(batch_size, -1, code_size // 3, 3)
        return batch_kp_preds

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
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg_raw = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        rcnn_reg, rcnn_reg_kp = rcnn_reg_raw[:, :self.box_coder.code_size], rcnn_reg_raw[:, self.box_coder.code_size:]
        # # Replace keypoint center with the box center
        # rcnn_reg_kp[:, -3:] = rcnn_reg[:, :3]

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_kp_preds = self.generate_predicted_keypoints(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], kp_preds=rcnn_reg_kp
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
