import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from functools import partial

from .center_head import CenterHead, SeparateHead


def decode_bbox_and_kp_bbox_from_heatmap(
        heatmap, rot_cos, rot_sin, center, center_z, dim,
        kp_rot_cos, kp_rot_sin, kp_center, kp_center_z, kp_dim,
        point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, iou=None, K=100,
        circle_nms=False, score_thresh=None, post_center_limit_range=None
):
    batch_size, num_class, _, _ = heatmap.size()

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, ys, xs = centernet_utils._topk(heatmap, K=K)
    # box 1
    center = centernet_utils._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    rot_sin = centernet_utils._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = centernet_utils._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
    center_z = centernet_utils._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = centernet_utils._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]

    # box 2
    center_kp = centernet_utils._transpose_and_gather_feat(kp_center, inds).view(batch_size, K, 2)
    rot_sin_kp = centernet_utils._transpose_and_gather_feat(kp_rot_sin, inds).view(batch_size, K, 1)
    rot_cos_kp = centernet_utils._transpose_and_gather_feat(kp_rot_cos, inds).view(batch_size, K, 1)
    center_z_kp = centernet_utils._transpose_and_gather_feat(kp_center_z, inds).view(batch_size, K, 1)
    dim_kp = centernet_utils._transpose_and_gather_feat(kp_dim, inds).view(batch_size, K, 3)

    angle_kp = torch.atan2(rot_sin_kp, rot_cos_kp)

    xs_kp = xs.view(batch_size, K, 1) + center_kp[:, :, 0:1]
    ys_kp = ys.view(batch_size, K, 1) + center_kp[:, :, 1:2]

    xs_kp = xs_kp * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys_kp = ys_kp * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    kp_box_part_list = [xs_kp, ys_kp, center_z_kp, dim_kp, angle_kp]
    if vel is not None:
        vel = centernet_utils._transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)
        kp_box_part_list.append(vel)

    if iou is not None:
        iou = centernet_utils._transpose_and_gather_feat(iou, inds).view(batch_size, K)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_kp_box_preds = torch.cat((kp_box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_kp_boxes = final_kp_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_kp_boxes': cur_kp_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })

        if iou is not None:
            ret_pred_dicts[-1]['pred_iou'] = iou[k, cur_mask]
    return ret_pred_dicts


class CenterHeadKPNaive(CenterHead):

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
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_kp_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.KP_HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            reg_kp_loss = self.reg_loss_func(
                pred_kp_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loc_kp_loss = (reg_kp_loss * reg_kp_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_kp_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + (loc_loss + loc_kp_loss) / 2
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            tb_dict['loc_kp_loss_head_%d' % idx] = loc_kp_loss.item()

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

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_kp_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)

            batch_kp_center = pred_dict['kp_center']
            batch_kp_center_z = pred_dict['kp_center_z']
            batch_kp_dim = pred_dict['kp_dim'].exp()
            batch_kp_rot_cos = pred_dict['kp_rot'][:, 0].unsqueeze(dim=1)
            batch_kp_rot_sin = pred_dict['kp_rot'][:, 1].unsqueeze(dim=1)

            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = decode_bbox_and_kp_bbox_from_heatmap(
                heatmap=batch_hm,
                rot_cos=batch_rot_cos, rot_sin=batch_rot_sin, center=batch_center, center_z=batch_center_z, dim=batch_dim,
                kp_rot_cos=batch_kp_rot_cos, kp_rot_sin=batch_kp_rot_sin, kp_center=batch_kp_center, kp_center_z=batch_kp_center_z, kp_dim=batch_kp_dim,
                vel=batch_vel, iou=batch_iou,
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
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_kp_boxes'] = final_dict['pred_kp_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_kp_boxes'].append(final_dict['pred_kp_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_kp_boxes'] = torch.cat(ret_dict[k]['pred_kp_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']
        pred_kp_boxes = pred_dicts[0]['pred_kp_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        rois_kp = pred_boxes.new_zeros((batch_size, num_max_rois, pred_kp_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            rois_kp[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_kp_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, rois_kp, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, rois_kp, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['rois_kp'] = rois_kp
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict