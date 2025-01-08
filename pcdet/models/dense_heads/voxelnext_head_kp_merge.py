from copy import deepcopy
import torch
from ..model_utils import model_nms_utils

from .voxelnext_head_kp import VoxelNeXtHeadKP, decode_bbox_and_keypoints_from_voxels_nuscenes


class VoxelNeXtHeadKPMerge(VoxelNeXtHeadKP):

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = torch.cat([pred_dict["loc_x"][..., 0:1], pred_dict["loc_y"][..., 0:1]], dim=-1)
        center_z = pred_dict["loc_z"][..., 0:1]
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box

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
                    pred_dict['loc_x'][bs_indices, 1] = - pred_dict['loc_x'][bs_indices, 1]
                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['vel'][bs_indices, 1] *= -1

                if i == 2:
                    pred_dict['loc_y'][bs_indices, 0] = - pred_dict['loc_y'][bs_indices, 0]
                    pred_dict['rot'][bs_indices, 0] *= -1
                    pred_dict['vel'][bs_indices, 0] *= -1

                if i == 3:
                    pred_dict['loc_x'][bs_indices, 0] = - pred_dict['loc_x'][bs_indices, 0]
                    pred_dict['loc_y'][bs_indices, 1] = - pred_dict['loc_y'][bs_indices, 1]

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
        batch_center = torch.cat([pred_dict["loc_x"][..., 0:1], pred_dict["loc_y"][..., 0:1]], dim=-1)
        batch_center_z = pred_dict_['loc_z'][:, 0:1]
        batch_dim = pred_dict_['dim']
        batch_rot_cos = pred_dict_['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict_['rot'][:, 1].unsqueeze(dim=1)
        batch_vel = pred_dict_['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
        batch_kp_x = pred_dict_['loc_x'][:, 1:]
        batch_kp_y = pred_dict_['loc_y'][:, 1:]
        batch_kp_z = pred_dict_['loc_z'][:, 1:]
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
        if not self.iou_branch:
            nms_config = deepcopy(post_process_cfg.NMS_CONFIG)
            nms_config.NMS_PRE_MAXSIZE = sum(post_process_cfg.NMS_CONFIG.NMS_PRE_MAXSIZE)
            nms_config.NMS_POST_MAXSIZE = sum(post_process_cfg.NMS_CONFIG.NMS_POST_MAXSIZE)
            nms_config.NMS_THRESH = post_process_cfg.NMS_CONFIG.NMS_THRESH[0]

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
                batch_hm = pred_dict['hm'].sigmoid()[:, 0:1]
                batch_center = torch.cat([pred_dict["loc_x"][..., 0:1], pred_dict["loc_y"][..., 0:1]], dim=-1)
                batch_center_z = pred_dict["loc_z"][..., 0:1]
                batch_dim = pred_dict['dim'].exp()
                batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
                batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
                batch_iou = (pred_dict['iou'] + 1) * 0.5 if self.iou_branch else None
                batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                voxel_indices_ = voxel_indices
                batch_kp_x = pred_dict['loc_x'][:, 1:]
                batch_kp_y = pred_dict['loc_y'][:, 1:]
                batch_kp_z = pred_dict['loc_z'][:, 1:]
                batch_kp_vis = pred_dict['kp_vis']

            # NOTE: This is for an old mistaken version. REMOVE IT AFTERWARDS.
            # batch_hm = batch_hm[:, 0:1] * ((self.point_cloud_range[-1] - self.point_cloud_range[2]) / self.voxel_size[2] / 8).ceil()
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
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if not self.iou_branch:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=nms_config,
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

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']

        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index

        # TODO: Merge Pedestrian and Cyclist into one class
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

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
