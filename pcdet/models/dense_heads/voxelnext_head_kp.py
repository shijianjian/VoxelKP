import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
from ...utils import loss_utils_kp
from ...utils.spconv_utils import replace_feature, spconv
import copy
from easydict import EasyDict

from .voxelnext_head import VoxelNeXtHead, SeparateHead


def decode_bbox_and_keypoints_from_voxels_nuscenes(
    batch_size, indices, obj, rot_cos, rot_sin,
    keypoints_x, keypoints_y, keypoints_z, keypoints_vis, num_keypoints,
    center, center_z, dim, vel=None, iou=None, point_cloud_range=None, voxel_size=None, voxels_3d=None,
    feature_map_stride=None, K=100, score_thresh=None, post_center_limit_range=None, add_features=None
):
    batch_idx = indices[:, 0]
    spatial_indices = indices[:, 1:]
    scores, inds, class_ids = centernet_utils._topk_1d(None, batch_size, batch_idx, obj, K=K, nuscenes=True)

    center = centernet_utils.gather_feat_idx(center, inds, batch_size, batch_idx)
    rot_sin = centernet_utils.gather_feat_idx(rot_sin, inds, batch_size, batch_idx)
    rot_cos = centernet_utils.gather_feat_idx(rot_cos, inds, batch_size, batch_idx)
    center_z = centernet_utils.gather_feat_idx(center_z, inds, batch_size, batch_idx)
    dim = centernet_utils.gather_feat_idx(dim, inds, batch_size, batch_idx)
    spatial_indices = centernet_utils.gather_feat_idx(spatial_indices, inds, batch_size, batch_idx)

    if not add_features is None:
        add_features = [centernet_utils.gather_feat_idx(add_feature, inds, batch_size, batch_idx) for add_feature in add_features]

    if not isinstance(feature_map_stride, int):
        feature_map_stride = centernet_utils.gather_feat_idx(feature_map_stride.unsqueeze(-1), inds, batch_size, batch_idx)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = (spatial_indices[:, :, -1:] + center[:, :, 0:1]) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = (spatial_indices[:, :, -2:-1] + center[:, :, 1:2]) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    # zs = (spatial_indices[:, :, 0:1]) * feature_map_stride * voxel_size[2] + point_cloud_range[2] + center_z

    box_part_list = [xs, ys, center_z, dim, angle]

    if not vel is None:
        vel = centernet_utils.gather_feat_idx(vel, inds, batch_size, batch_idx)
        box_part_list.append(vel)

    # keypoints
    kp_x = centernet_utils.gather_feat_idx(keypoints_x, inds, batch_size, batch_idx)
    kp_y = centernet_utils.gather_feat_idx(keypoints_y, inds, batch_size, batch_idx)
    kp_z = centernet_utils.gather_feat_idx(keypoints_z, inds, batch_size, batch_idx)
    kp_vis = centernet_utils.gather_feat_idx(keypoints_vis, inds, batch_size, batch_idx)

    kp_x = (spatial_indices[:, :, -1:] + kp_x) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    kp_y = (spatial_indices[:, :, -2:-1] + kp_y) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    # kp_x = xs + kp_x
    # kp_y = ys + kp_y

    kp_part_list = [kp_x, kp_y, kp_z, kp_vis]

    if not iou is None:
        iou = centernet_utils.gather_feat_idx(iou, inds, batch_size, batch_idx)
        iou = torch.clamp(iou, min=0, max=1.)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_kp_preds = torch.stack((kp_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)

    final_class_ids = class_ids.view(batch_size, K)
    if not add_features is None:
        add_features = [add_feature.view(batch_size, K, add_feature.shape[-1]) for add_feature in add_features]

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_kps = final_kp_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]
        cur_add_features = [add_feature[k, cur_mask] for add_feature in add_features] if not add_features is None else None
        cur_iou = iou[k, cur_mask] if not iou is None else None

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_kps': cur_kps[..., :3],
            'pred_kps_vis': cur_kps[..., 3:4],
            'pred_scores': cur_scores,
            'pred_labels': cur_labels,
            'pred_ious': cur_iou,
            'add_features': cur_add_features,
        })
    return ret_pred_dicts


class VoxelNeXtHeadKP(VoxelNeXtHead):

    # 0  KeypointType.KEYPOINT_TYPE_NOSE,
    # 1  KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER,
    # 2  KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER,
    # 3  KeypointType.KEYPOINT_TYPE_LEFT_ELBOW,
    # 4  KeypointType.KEYPOINT_TYPE_RIGHT_ELBOW,
    # 5  KeypointType.KEYPOINT_TYPE_LEFT_WRIST,
    # 6  KeypointType.KEYPOINT_TYPE_RIGHT_WRIST,
    # 7  KeypointType.KEYPOINT_TYPE_LEFT_HIP,
    # 8  KeypointType.KEYPOINT_TYPE_RIGHT_HIP,
    # 9  KeypointType.KEYPOINT_TYPE_LEFT_KNEE,
    # 10 KeypointType.KEYPOINT_TYPE_RIGHT_KNEE,
    # 11 KeypointType.KEYPOINT_TYPE_LEFT_ANKLE,
    # 12 KeypointType.KEYPOINT_TYPE_RIGHT_ANKLE,
    # 13 KeypointType.KEYPOINT_TYPE_HEAD_CENTER

    # joints_a = [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 13, 1, 2, 1, 7]
    # joints_b = [13, 3, 4, 5, 6, 9, 10, 11, 12, 1, 2, 7, 8, 2, 8]

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=False):
        class_name_map = model_cfg.get("MAP_CLASS_NAMES")
        out_classes = list(set([class_name_map[cls_name] for cls_name in class_names]))
        super().__init__(model_cfg, input_channels, len(out_classes), out_classes, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training)

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())
        self.add_module('bone_loss_func', loss_utils_kp.BoneLossCenterNet(
            [ 0, 1, 2, 3, 4, 7,  8,  9, 10, 13, 13, 1, 2],
            [13, 3, 4, 5, 6, 9, 10, 11, 12,  1,  2, 7, 8],
        ))
        if self.iou_branch:
            self.add_module('crit_iou', loss_utils.IouLossSparse())
            self.add_module('crit_iou_reg', loss_utils.IouRegLossSparse())

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, gt_keypoints, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            gt_keypoints: (N, N, 3 + 1)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)

        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        # Keypoints part
        ret_kps = gt_boxes.new_zeros((num_max_objs, gt_keypoints.shape[-2], gt_keypoints.shape[-1]))

        keypoint_x, keypoint_y, keypoint_z, keypoint_vis = gt_keypoints[..., 0], gt_keypoints[..., 1], gt_keypoints[..., 2], gt_keypoints[..., 3]

        dx_kp = (keypoint_x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        dy_kp = (keypoint_y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)

            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)

            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

            # Keypoints part
            # TODO: use .log()
            ret_kps[k, :, 0] = dx_kp[k] - spatial_indices[inds[k]][0]
            ret_kps[k, :, 1] = dy_kp[k] - spatial_indices[inds[k]][1]
            ret_kps[k, :, 2] = keypoint_z[k]
            ret_kps[k, :, 3] = keypoint_vis[k]

        return heatmap, ret_boxes, ret_kps, inds, mask

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['voxel_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            target_kps = target_dicts['target_kps'][idx]
            target_kps_vis = target_dicts['target_kps_vis'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            # torch.Size([66248, 8]
            pred_kp_x = pred_dict["kp_x"]
            pred_kp_y = pred_dict["kp_y"]
            pred_kp_z = pred_dict["kp_z"]
            pred_kp_vis = pred_dict["kp_vis"]
            # torch.Size([66248, 14, 3])

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )

            bone_loss = self.bone_loss_func(
                torch.stack([pred_kp_x, pred_kp_y, pred_kp_z], dim=-1),
                target_dicts['masks'][idx], target_dicts['inds'][idx], target_kps, batch_index
            )
            reg_kp_vis_loss = self.reg_loss_func(
                pred_kp_vis, target_dicts['masks'][idx], target_dicts['inds'][idx], target_kps_vis, batch_index
            )

            reg_kp_x_loss = self.reg_loss_func(
                pred_kp_x, target_dicts['masks'][idx], target_dicts['inds'][idx], target_kps[..., 0], batch_index
            )
            reg_kp_y_loss = self.reg_loss_func(
                pred_kp_y, target_dicts['masks'][idx], target_dicts['inds'][idx], target_kps[..., 1], batch_index
            )
            reg_kp_z_loss = self.reg_loss_func(
                pred_kp_z, target_dicts['masks'][idx], target_dicts['inds'][idx], target_kps[..., 2], batch_index
            )

            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            reg_kp_vis_loss = (reg_kp_vis_loss * reg_kp_vis_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_visibility_code_weights'])).mean()
            reg_kp_vis_loss = reg_kp_vis_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_visibility_weights']

            bone_loss = bone_loss.mean() * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_visibility_weights']

            reg_kp_x_loss = (reg_kp_x_loss * reg_kp_x_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_x_code_weights'])).sum()
            reg_kp_x_loss = reg_kp_x_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_x_loc_weight']

            reg_kp_y_loss = (reg_kp_y_loss * reg_kp_y_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_y_code_weights'])).sum()
            reg_kp_y_loss = reg_kp_y_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_y_loc_weight']

            reg_kp_z_loss = (reg_kp_z_loss * reg_kp_z_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_z_code_weights'])).sum()
            reg_kp_z_loss = reg_kp_z_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kp_z_loc_weight']

            reg_kp_loss = (reg_kp_x_loss + reg_kp_y_loss + reg_kp_z_loss) / 3

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            tb_dict['loc_kp_x_loss_head_%d' % idx] = reg_kp_x_loss.item()
            tb_dict['loc_kp_y_loss_head_%d' % idx] = reg_kp_y_loss.item()
            tb_dict['loc_kp_z_loss_head_%d' % idx] = reg_kp_z_loss.item()
            tb_dict['loc_kp_all_loss_head_%d' % idx] = reg_kp_loss.item()
            tb_dict['kp_vis_loss_head_%d' % idx] = reg_kp_vis_loss.item()
            tb_dict['kp_bone_loss_head_%d' % idx] = bone_loss.item()

            loss = bone_loss
            if self.iou_branch:
                batch_box_preds = self._get_predicted_boxes(pred_dict, spatial_indices)
                pred_boxes_for_iou = batch_box_preds.detach()
                iou_loss = self.crit_iou(pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx],
                                            pred_boxes_for_iou, target_dicts['gt_boxes'][idx], batch_indices)

                iou_reg_loss = self.crit_iou_reg(batch_box_preds, target_dicts['masks'][idx], target_dicts['inds'][idx],
                                                    target_dicts['gt_boxes'][idx], batch_indices)
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] if 'iou_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS else self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                iou_reg_loss = iou_reg_loss * iou_weight  # self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

                loss += (hm_loss + (loc_loss + reg_kp_loss) / 2 + reg_kp_vis_loss + iou_loss + iou_reg_loss)
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            else:
                loss += hm_loss + (loc_loss + reg_kp_loss) / 2 + reg_kp_vis_loss

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = pred_dict['center']
        center_z = pred_dict['center_z']
        #dim = pred_dict['dim'].exp()
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box

    def rotate_class_specific_nms_iou(self, boxes, kps, scores, iou_preds, labels, rectifier, nms_configs):
        """
        :param boxes: (N, 5) [x, y, z, l, w, h, theta]
        :param kps: (N, 14, 3)
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert isinstance(rectifier, list)

        box_preds_list, kps_pred_list, scores_list, labels_list = [], [], [], []
        for cls in range(self.num_class):
            mask = labels == cls
            boxes_cls = boxes[mask]
            kps_cls = kps[mask]
            scores_cls = torch.pow(scores[mask], 1 - rectifier[cls]) * torch.pow(iou_preds[mask].squeeze(-1), rectifier[cls])
            labels_cls = labels[mask]

            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=scores_cls, box_preds=boxes_cls, nms_config=nms_configs[cls], score_thresh=None)

            box_preds_list.append(boxes_cls[selected])
            kps_pred_list.append(kps_cls[selected])
            scores_list.append(scores_cls[selected])
            labels_list.append(labels_cls[selected])

        return torch.cat(box_preds_list, dim=0), torch.cat(kps_pred_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)

    def merge_double_flip(self, pred_dict, batch_size, voxel_indices, spatial_shape):
        # spatial_shape (Z, Y, X)
        pred_dict['hm'] = pred_dict['hm'].sigmoid()
        pred_dict['dim'] = pred_dict['dim'].exp()

        batch_indices = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]

        pred_dict_ = {k: [] for k in pred_dict.keys()}
        counts = []
        spatial_indices_ = []
        for bs_idx in range(batch_size):
            spatial_indices_batch = []
            pred_dict_batch = {k: [] for k in pred_dict.keys()}
            for i in range(4):
                bs_indices = batch_indices == (bs_idx * 4 + i)
                if i in [1, 3]:
                    spatial_indices[bs_indices, 0] = spatial_shape[0] - spatial_indices[bs_indices, 0]
                if i in [2, 3]:
                    spatial_indices[bs_indices, 1] = spatial_shape[1] - spatial_indices[bs_indices, 1]

                if i == 1:
                    pred_dict['center'][bs_indices, 1] = - pred_dict['center'][bs_indices, 1]
                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['vel'][bs_indices, 1] *= -1

                if i == 2:
                    pred_dict['center'][bs_indices, 0] = - pred_dict['center'][bs_indices, 0]
                    pred_dict['rot'][bs_indices, 0] *= -1
                    pred_dict['vel'][bs_indices, 0] *= -1

                if i == 3:
                    pred_dict['center'][bs_indices, 0] = - pred_dict['center'][bs_indices, 0]
                    pred_dict['center'][bs_indices, 1] = - pred_dict['center'][bs_indices, 1]

                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['rot'][bs_indices, 0] *= -1

                    pred_dict['vel'][bs_indices] *= -1

                spatial_indices_batch.append(spatial_indices[bs_indices])

                for k in pred_dict.keys():
                    pred_dict_batch[k].append(pred_dict[k][bs_indices])

            spatial_indices_batch = torch.cat(spatial_indices_batch)

            spatial_indices_unique, _inv, count = torch.unique(spatial_indices_batch, dim=0, return_inverse=True,
                                                               return_counts=True)
            spatial_indices_.append(spatial_indices_unique)
            counts.append(count)
            for k in pred_dict.keys():
                pred_dict_batch[k] = torch.cat(pred_dict_batch[k])
                features_unique = pred_dict_batch[k].new_zeros(
                    (spatial_indices_unique.shape[0], pred_dict_batch[k].shape[1]))
                features_unique.index_add_(0, _inv, pred_dict_batch[k])
                pred_dict_[k].append(features_unique)

        for k in pred_dict.keys():
            pred_dict_[k] = torch.cat(pred_dict_[k])
        counts = torch.cat(counts).unsqueeze(-1).float()
        voxel_indices_ = torch.cat([torch.cat(
            [torch.full((indices.shape[0], 1), i, device=indices.device, dtype=indices.dtype), indices], dim=1
        ) for i, indices in enumerate(spatial_indices_)])

        batch_hm = pred_dict_['hm']
        batch_center = pred_dict_['center']
        batch_center_z = pred_dict_['center_z']
        batch_dim = pred_dict_['dim']
        batch_rot_cos = pred_dict_['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict_['rot'][:, 1].unsqueeze(dim=1)
        batch_vel = pred_dict_['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
        batch_kp_x = pred_dict_['kp_x']
        batch_kp_y = pred_dict_['kp_y']
        batch_kp_z = pred_dict_['kp_z']
        batch_kp_vis = pred_dict_['kp_vis']

        batch_hm /= counts
        batch_center /= counts
        batch_center_z /= counts
        batch_dim /= counts
        batch_rot_cos /= counts
        batch_rot_sin /= counts
        batch_kp_x /= counts
        batch_kp_y /= counts
        batch_kp_z /= counts
        batch_kp_vis /= counts

        if not batch_vel is None:
            batch_vel /= counts

        return batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, None, voxel_indices_, batch_kp_x, batch_kp_y, batch_kp_z, batch_kp_vis

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_kps': [],
            'pred_kps_vis': [],
            'pred_scores': [],
            'pred_labels': [],
            'pred_ious': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            if self.double_flip:
                batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, batch_iou, voxel_indices_, batch_kp_x, batch_kp_y, batch_kp_z, batch_kp_vis = \
                self.merge_double_flip(pred_dict, batch_size, voxel_indices.clone(), spatial_shape)
            else:
                batch_hm = pred_dict['hm'].sigmoid()
                batch_center = pred_dict['center']
                batch_center_z = pred_dict['center_z']
                batch_dim = pred_dict['dim'].exp()
                batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
                batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
                batch_iou = (pred_dict['iou'] + 1) * 0.5 if self.iou_branch else None
                batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                voxel_indices_ = voxel_indices
                batch_kp_x = pred_dict['kp_x']
                batch_kp_y = pred_dict['kp_y']
                batch_kp_z = pred_dict['kp_z']
                batch_kp_vis = pred_dict['kp_vis']

            final_pred_dicts = decode_bbox_and_keypoints_from_voxels_nuscenes(
                batch_size=batch_size, indices=voxel_indices_,
                obj=batch_hm, 
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                keypoints_x=batch_kp_x, keypoints_y=batch_kp_y, keypoints_z=batch_kp_z, keypoints_vis=batch_kp_vis, num_keypoints=batch_kp_y.size(1),
                center=batch_center, center_z=batch_center_z,
                dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                #circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                try:
                    final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                except Exception as e:
                    assert False, (final_dict['pred_labels'])
                if not self.iou_branch:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_kps'] = final_dict['pred_kps'][selected]
                    final_dict['pred_kps_vis'] = final_dict['pred_kps_vis'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_kps'].append(final_dict['pred_kps'])
                ret_dict[k]['pred_kps_vis'].append(final_dict['pred_kps_vis'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
                ret_dict[k]['pred_ious'].append(final_dict['pred_ious'])

        for k in range(batch_size):
            pred_boxes = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            pred_kps = torch.cat([torch.cat(ret_dict[k]['pred_kps'], dim=0), torch.cat(ret_dict[k]['pred_kps_vis'], dim=0)], dim=-1)
            pred_scores = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            pred_labels = torch.cat(ret_dict[k]['pred_labels'], dim=0)
            if self.iou_branch:
                pred_ious = torch.cat(ret_dict[k]['pred_ious'], dim=0)
                pred_boxes, pred_kps, pred_scores, pred_labels = self.rotate_class_specific_nms_iou(
                    pred_boxes, pred_kps, pred_scores, pred_ious, pred_labels, self.rectifier, self.nms_configs)

            ret_dict[k]['pred_boxes'] = pred_boxes
            ret_dict[k]['pred_kps'] = pred_kps[..., :3]
            ret_dict[k]['pred_kps_vis'] = pred_kps[..., 3]
            ret_dict[k]['pred_scores'] = pred_scores
            ret_dict[k]['pred_labels'] = pred_labels + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        raise NotImplementedError
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
            rois_kp[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_kps']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, rois_kp, roi_scores, roi_labels

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']

        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index

        # TODO: Merge Pedestrian and Cyclist into one class
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], data_dict['keypoint_location'], data_dict['keypoint_visibility'], num_voxels, spatial_indices, spatial_shape
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['voxel_indices'] = voxel_indices

        if not self.training or self.predict_boxes_when_training:
            if self.double_flip:
                data_dict['batch_size'] = data_dict['batch_size'] // 4
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'],
                pred_dicts, voxel_indices, spatial_shape
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
