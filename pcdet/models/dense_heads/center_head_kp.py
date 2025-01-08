import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils

from .center_head import CenterHead


def get_box_from_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Args:
        keypoints: BxNx42, the last three points are the center.
    """
    assert len(keypoints.shape) == 3 and keypoints.shape[-1] == 42
    batchsize = keypoints.shape[0]
    kp_center = keypoints[..., -3:]
    kp_range = keypoints.view(batchsize, -1, 14, 3).amax(dim=-2) - keypoints.view(batchsize, -1, 14, 3).amin(dim=-2)
    pred_boxes = torch.cat([kp_center, kp_range], dim=-1)
    return pred_boxes


class KeypointSelfEncoder(object):
    """Make points relative to the box center and normalize against box length.

    Args:
        num_joints: 13 joints plus 1 faked center point.
    """
    def __init__(self, num_joints=14, **kwargs):
        super().__init__()
        self.code_size = num_joints * 3

    def encode(self, keypoints):
        """
        Args:
            keypoints: (B, N, K, 3) [x, y, z, ..., center_x, center_y, center_z]

        Note:
            center_x, center_y, center_z may be any 3D point.
        """
        center_x, center_y = keypoints[..., -1:, 0], keypoints[..., -1:, 1]
        # center_x, center_y, center_z = keypoints[..., -1:, 0], keypoints[..., -1:, 1], keypoints[..., -1:, 2]

        dxt = keypoints[..., 0]
        dyt = keypoints[..., 1]
        dxt[..., :13] = dxt[..., :13] - center_x
        dyt[..., :13] = dyt[..., :13] - center_y
        dzt = keypoints[..., :, 2]

        return torch.stack([dxt, dyt, dzt], dim=-1)

    def decode(self, keypoint_encodings):
        """
        Args:
            keypoint_encodings: (B, N, K, 3) [x, y, z, ..., center_x, center_y, center_z]

        Note:
            center_x, center_y, center_z may be any 3D point.
        """

        center_x, center_y = keypoint_encodings[..., -1:, 0], keypoint_encodings[..., -1:, 1]
        # center_x, center_y, center_z = keypoints[..., -1:, 0], keypoints[..., -1:, 1], keypoints[..., -1:, 2]

        xt = keypoint_encodings[..., 0]
        yt = keypoint_encodings[..., 1]
        xt[..., :13] = xt[..., :13] + center_x
        yt[..., :13] = yt[..., :13] + center_y
        zt = keypoint_encodings[..., :, 2]

        return torch.stack([xt, yt, zt], dim=-1)



class CenterHeadKP(CenterHead):

    kp_coder = KeypointSelfEncoder()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_keypoints = torch.stack([pred_dict[head_name] for head_name in self.separate_head_cfg.KP_HEAD_ORDER], dim=-1)
            # torch.Size([4, 13, 188, 188, 3])
            pred_center = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.CENTER_ORDER], dim=1)
            # torch.Size([4, 3, 188, 188])
            assert False, (target_boxes.shape, pred_center.shape)

            reg_loss = self.reg_loss_func(
                pred_keypoints, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):

                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                    pred_dict=pred_dict,
                    point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride
                )  # (B, H, W, 7 or 9)

                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                    iou_loss = loss_utils.calculate_iou_loss_centerhead(
                        iou_preds=pred_dict['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_keypoints(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_keypoints': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            # the center
            batch_hm = pred_dict['hm'].sigmoid()
            batch_kp_x = pred_dict['kp_x']
            batch_kp_y = pred_dict['kp_y']
            batch_kp_z = pred_dict['kp_z']
            # batch_center = pred_dict['center']
            # batch_center_z = pred_dict['center_z']
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.KP_HEAD_ORDER else None

            # The last dim as center
            # batch_center = torch.stack(batch_kp_x[..., -3], batch_kp_y[..., -2], dim=-1)
            # batch_center_z = batch_kp_z[..., -1][..., None]

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_keypoints_from_heatmap(
                heatmap=batch_hm, offset_kp_x=batch_kp_x, offset_kp_y=batch_kp_y,
                offset_kp_z=batch_kp_z, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:

                    pred_kp = self.kp_coder.decode(final_dict['pred_keypoints'])
                    pred_boxes = get_box_from_keypoints(pred_kp)

                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=pred_boxes,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':

                    pred_kp = self.kp_coder.decode(final_dict['pred_keypoints'])
                    pred_boxes = get_box_from_keypoints(pred_kp)

                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=pred_boxes,
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_keypoints'] = final_dict['pred_keypoints'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_keypoints'].append(final_dict['pred_keypoints'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_keypoints'] = torch.cat(ret_dict[k]['pred_keypoints'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_keypoints']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_keypoints = pred_dicts[0]['pred_keypoints']

        rois = pred_keypoints.new_zeros((batch_size, num_max_rois, pred_keypoints.shape[-1]))
        roi_scores = pred_keypoints.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_keypoints.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_keypoints'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_keypoints']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        # Set keypoint center as box center
        data_dict['keypoint_location'][0, ..., -1, :] = data_dict['gt_boxes'][0, ..., :3]
        # assert False, (get_box_from_keypoints(data_dict['keypoint_location'].view(4, -1, 42))[0], data_dict['gt_boxes'][0])
        data_dict['keypoint_location'] = self.kp_coder.encode(data_dict['keypoint_location'])

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_keypoints(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
