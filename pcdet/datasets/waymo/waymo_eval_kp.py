import numpy as np
import torch
import pickle
import scipy
import tensorflow as tf
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.metrics.python import keypoint_metrics
# Import as the new version of keypoint_metrics
# from . import keypoint_metrics
import argparse

from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator as _est
from ...utils import box_utils
from ...utils import common_utils


tf.get_logger().setLevel('INFO')


def create_combined_metric(config: keypoint_metrics.CombinedMetricsConfig) -> keypoint_metrics.KeypointsMetric:
  """Creates a combined metric with all keypoint metrics.

  Args:
    config: See `CombinedMetricsConfig` for details. Use `DEFAULT_CONFIG_*`
      constants for default configurations to get a full set of all supported
      metrics.

  Returns:
    a keypoint metric which is a subclass of `tf.keras.metrics.Metric`, see
    `KeypointsMetric` for details.
  """
  def _create_pem(subset: str) -> keypoint_metrics.PoseEstimationMetric:
    return keypoint_metrics.PoseEstimationMetric(0.25, f"PEM/{subset}")

  pem = keypoint_metrics.MetricForSubsets(
      src_order=config.src_order,
      subsets=config.subsets,
      create_metric_fn=_create_pem,
  )
  return keypoint_metrics.CombinedMetric([pem])


class OpenPCDetWaymoDetectionMetricsEstimator(_est):

    def waymo_evaluation(self, prediction_infos, gt_infos, class_name, distance_thresh=100, fake_gt_infos=True, has_kp=True):
        detection_aps = super().waymo_evaluation(prediction_infos, gt_infos, class_name, distance_thresh, fake_gt_infos, has_kp)

        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        pd_waymo = self.generate_waymo_type_results(
            prediction_infos, class_name, is_gt=False, has_kp=has_kp
        )
        gt_waymo = self.generate_waymo_type_results(
            gt_infos, class_name, is_gt=True, fake_gt_infos=fake_gt_infos, has_kp=has_kp
        )

        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _, pd_kp3d, pd_kp3d_vis = pd_waymo
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty, gt_kp3d, gt_kp3d_vis = gt_waymo

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz, pd_kp3d, pd_kp3d_vis = self.mask_by_distance(
            distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz, pd_kp3d, pd_kp3d_vis)
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty, gt_kp3d, gt_kp3d_vis = self.mask_by_distance(
            distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty, gt_kp3d, gt_kp3d_vis)

        kp_evals = self.evaluate_keypoints(pd_kp3d, pd_kp3d_vis, pd_boxes3d, gt_kp3d, gt_kp3d_vis, gt_boxes3d, pd_frameid, gt_frameid)
        return detection_aps, kp_evals

    def evaluate_keypoints(self, pd_kp3d, pd_kp3d_vis, pd_boxes, gt_kp3d, gt_kp3d_vis, gt_boxes, pd_frameid, gt_frameid, IOU_THRESH=0.5):
        """
        args:
            IOU_THRESH: 0. means we do not exclude any boxes from the ground truth due to no valid match.
        """

        def find_best_match_by_boxes(pd_boxes, gt_boxes, IOU_THRESH):
            # ref: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
            pd_boxes = torch.as_tensor(pd_boxes)
            gt_boxes = torch.as_tensor(gt_boxes)
            # pd_size x gt_size
            iou_matrix = box_utils.boxes3d_nearest_bev_iou(gt_boxes[:, 0:7], pd_boxes[:, 0:7]).numpy()
            min_iou = 0.
            if pd_boxes.size(0) > gt_boxes.size(0):
                # there are more predictions than ground-truth - add dummy rows
                diff = pd_boxes.size(0) - gt_boxes.size(0)
                iou_matrix = np.concatenate((iou_matrix, np.full((diff, pd_boxes.size(0)), min_iou)), axis=0)
            if gt_boxes.size(0) > pd_boxes.size(0):
                # more ground-truth than predictions - add dummy columns
                diff = gt_boxes.size(0) - pd_boxes.size(0)
                iou_matrix = np.concatenate((iou_matrix, np.full((gt_boxes.size(0), diff), min_iou)), axis=1)

            # Hungarian matching
            # idxs_true, idxs_pred : indices into gt and pred for matches
            idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

            # remove dummy assignments
            sel_pred = idxs_pred < pd_boxes.size(0)
            idx_pred_actual = idxs_pred[sel_pred]
            idx_gt_actual = idxs_true[sel_pred]
            ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
            sel_valid = (ious_actual > IOU_THRESH)

            return idx_gt_actual, idx_pred_actual, ious_actual, sel_valid

        def find_best_match_by_keypoints(pd_keypoints, gt_keypoints, gt_kp3d_vis, IOU_THRESH):
            pd_keypoints = torch.as_tensor(pd_keypoints)
            gt_keypoints = torch.as_tensor(gt_keypoints)

            # Calculate L1 distances using broadcasting
            pd_keypoints_expanded = pd_keypoints.unsqueeze(0)  # Add a new dimension for broadcasting
            gt_keypoints_expanded = gt_keypoints.unsqueeze(1)  # Add a new dimension for broadcasting

            if gt_kp3d_vis is not None:
                gt_kp3d_vis = torch.as_tensor(gt_kp3d_vis)
                gt_kp3d_vis_expanded = gt_kp3d_vis.unsqueeze(1)
                l1_vis = torch.abs(gt_keypoints_expanded - pd_keypoints_expanded) * gt_kp3d_vis_expanded[..., None]
                l1_distances = torch.mean(l1_vis.mean(-1), dim=2)
            else:
                l1_distances = torch.sum(torch.abs(gt_keypoints_expanded - pd_keypoints_expanded).mean(-1), dim=2)  # Calculate L1 distances
            # Convert the L1 distances to a NumPy array for compatibility with linear_sum_assignment
            cost_matrix = l1_distances.cpu().numpy()  # Convert to NumPy array

            min_iou = 0.
            if pd_keypoints.size(0) > gt_keypoints.size(0):
                # there are more predictions than ground-truth - add dummy rows
                diff = pd_keypoints.size(0) - gt_keypoints.size(0)
                cost_matrix = np.concatenate((cost_matrix, np.full((diff, pd_keypoints.size(0)), min_iou)), axis=0)
            if gt_keypoints.size(0) > pd_keypoints.size(0):
                # more ground-truth than predictions - add dummy columns
                diff = gt_keypoints.size(0) - pd_keypoints.size(0)
                cost_matrix = np.concatenate((cost_matrix, np.full((gt_keypoints.size(0), diff), min_iou)), axis=1)

            # Hungarian matching
            # idxs_true, idxs_pred : indices into gt and pred for matches
            idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(cost_matrix)

            # remove dummy assignments
            sel_pred = idxs_pred < pd_keypoints.size(0)
            idx_pred_actual = idxs_pred[sel_pred]
            idx_gt_actual = idxs_true[sel_pred]
            ious_actual = cost_matrix[idx_gt_actual, idx_pred_actual]
            sel_valid = (ious_actual > IOU_THRESH)

            return idx_gt_actual, idx_pred_actual, ious_actual, sel_valid

        # def _zero_pad_to_size(np_array, target_size):  # For PEM
        #     zero_array = np.zeros(((target_size - np_array.shape[0]), *np_array.shape[1:]))
        #     return np.concatenate([np_array, zero_array], axis=0)
        # assert False, (pd_kp3d.shape, gt_kp3d.shape, gt_frameid.shape, pd_frameid.shape)

        def _alternative_matcher():
            # selected_gt_kp3d, selected_pd_kp3d, selected_pd_kp3d, selected_pd_kp3d = [], []
            selected_pd_kp3d, selected_pd_kp3d_vis, selected_pd_boxes, selected_gt_kp3d, selected_gt_kp3d_vis, selected_gt_boxes = [], [], [], [], [], []
            unpaired_pd_kp3d, unpaired_pd_kp3d_vis, unpaired_pd_boxes, unpaired_gt_kp3d, unpaired_gt_kp3d_vis, unpaired_gt_boxes = [], [], [], [], [], []
            for frame_id in np.unique(gt_frameid):
                # idx_gt_actual, idx_pred_actual, _, sel_valid = find_best_match_by_keypoints(pd_kp3d[pd_frameid == frame_id], gt_kp3d[gt_frameid == frame_id], gt_kp3d_vis[gt_frameid == frame_id], IOU_THRESH=IOU_THRESH)
                idx_gt_actual, idx_pred_actual, _, sel_valid = find_best_match_by_boxes(pd_boxes[pd_frameid == frame_id], gt_boxes[gt_frameid == frame_id], IOU_THRESH=IOU_THRESH)
                selected_pd_kp3d += [pd_kp3d[pd_frameid == frame_id][idx_pred_actual[sel_valid]]]
                selected_pd_kp3d_vis += [pd_kp3d_vis[pd_frameid == frame_id][idx_pred_actual[sel_valid]]]
                selected_pd_boxes += [pd_boxes[pd_frameid == frame_id][idx_pred_actual[sel_valid]]]
                selected_gt_kp3d += [gt_kp3d[gt_frameid == frame_id][idx_gt_actual[sel_valid]]]
                selected_gt_kp3d_vis += [gt_kp3d_vis[gt_frameid == frame_id][idx_gt_actual[sel_valid]]]
                selected_gt_boxes += [gt_boxes[gt_frameid == frame_id][idx_gt_actual[sel_valid]]]

                unpaired_pd_kp3d += [pd_kp3d[pd_frameid == frame_id][~idx_pred_actual[sel_valid]]]
                unpaired_pd_kp3d_vis += [pd_kp3d_vis[pd_frameid == frame_id][~idx_pred_actual[sel_valid]]]
                unpaired_pd_boxes += [pd_boxes[pd_frameid == frame_id][~idx_pred_actual[sel_valid]]]
                unpaired_gt_kp3d += [gt_kp3d[gt_frameid == frame_id][~idx_gt_actual[sel_valid]]]
                unpaired_gt_kp3d_vis += [gt_kp3d_vis[gt_frameid == frame_id][~idx_gt_actual[sel_valid]]]
                unpaired_gt_boxes += [gt_boxes[gt_frameid == frame_id][~idx_gt_actual[sel_valid]]]

            selected_pd_kp3d = np.concatenate(selected_pd_kp3d, axis=0)
            selected_pd_kp3d_vis = np.concatenate(selected_pd_kp3d_vis, axis=0)
            selected_pd_boxes = np.concatenate(selected_pd_boxes, axis=0)
            selected_gt_kp3d = np.concatenate(selected_gt_kp3d, axis=0)
            selected_gt_kp3d_vis = np.concatenate(selected_gt_kp3d_vis, axis=0)
            selected_gt_boxes = np.concatenate(selected_gt_boxes, axis=0)

            unpaired_pd_kp3d = np.concatenate(unpaired_pd_kp3d, axis=0)
            unpaired_pd_kp3d_vis = np.concatenate(unpaired_pd_kp3d_vis, axis=0)
            unpaired_pd_boxes = np.concatenate(unpaired_pd_boxes, axis=0)
            unpaired_gt_kp3d = np.concatenate(unpaired_gt_kp3d, axis=0)
            unpaired_gt_kp3d_vis = np.concatenate(unpaired_gt_kp3d_vis, axis=0)
            unpaired_gt_boxes = np.concatenate(unpaired_gt_boxes, axis=0)

            pred_kps = keypoint_data.KeypointsTensors(
                location=tf.convert_to_tensor(np.asarray(selected_pd_kp3d).astype(np.float32), dtype=np.float32),
                visibility=tf.convert_to_tensor(np.asarray(selected_pd_kp3d_vis).astype(np.float32), dtype=np.float32)
            )
            gt_kps = keypoint_data.KeypointsTensors(
                location=tf.convert_to_tensor(selected_gt_kp3d, dtype=tf.float32),
                visibility=tf.convert_to_tensor(selected_gt_kp3d_vis, dtype=tf.float32)
            )
            gt_boxTensor = keypoint_data.BoundingBoxTensors(
                center=tf.convert_to_tensor(selected_gt_boxes[..., :3], dtype=tf.float32),
                size=tf.convert_to_tensor(selected_gt_boxes[..., 3:6], dtype=tf.float32),
                heading=tf.convert_to_tensor(selected_gt_boxes[..., 6], dtype=tf.float32)
            )

            pem_pred_kps = keypoint_data.KeypointsTensors(
                location=tf.convert_to_tensor(np.concatenate([selected_pd_kp3d, np.zeros_like(unpaired_gt_kp3d), unpaired_pd_kp3d]), dtype=np.float32),
                visibility=tf.convert_to_tensor(np.concatenate([selected_pd_kp3d_vis, unpaired_pd_kp3d_vis, np.zeros_like(unpaired_gt_kp3d_vis)]), dtype=np.float32)
            )
            pem_gt_kps = keypoint_data.KeypointsTensors(
                location=tf.convert_to_tensor(np.concatenate([selected_gt_kp3d, unpaired_gt_kp3d, np.zeros_like(unpaired_pd_kp3d)]), dtype=tf.float32),
                visibility=tf.convert_to_tensor(np.concatenate([selected_gt_kp3d_vis, np.zeros_like(unpaired_pd_kp3d_vis), unpaired_gt_kp3d_vis]), dtype=tf.float32)
            )
            pem_gt_boxTensor = keypoint_data.BoundingBoxTensors(
                center=tf.convert_to_tensor(np.concatenate([
                    selected_gt_boxes[..., :3], np.zeros_like(unpaired_gt_boxes[..., :3]), unpaired_pd_boxes[..., :3]]), dtype=tf.float32),
                size=tf.convert_to_tensor(np.concatenate([
                    selected_gt_boxes[..., 3:6], np.zeros_like(unpaired_gt_boxes[..., 3:6]), unpaired_pd_boxes[..., 3:6]]), dtype=tf.float32),
                heading=tf.convert_to_tensor(np.concatenate([
                    selected_gt_boxes[..., 6], np.zeros_like(unpaired_gt_boxes[..., 6]), unpaired_pd_boxes[..., 6]]), dtype=tf.float32),
            )

            return (
                pred_kps, None, gt_kps, gt_boxTensor,
                pem_pred_kps, None, pem_gt_kps, pem_gt_boxTensor
            )

        def _offical_matcher():

            def find_matched_only(matcher, pose):
                """Return only matched.
                """
                assert pose.gt.keypoints is not None
                assert pose.gt.keypoints.has_visible is not None
                # Indices of both GTi and GTv matching with PR.
                m_ids = matcher.matching_ids(pose)
                m_visible_ids = keypoint_metrics._select_ids(m_ids, pose.gt.keypoints.has_visible)

                # Output ground truth order: matched objects, false negatives, padding.
                gt_m = keypoint_metrics._reorder_objects(pose.gt, m_visible_ids.gt)
                # Output predictions order: matched objects, padding, false positives.
                pr_m = keypoint_metrics._reorder_objects(pose.pr, m_visible_ids.pr)
                return keypoint_metrics.PoseEstimationPair(gt_m, pr_m)

            matcher = keypoint_metrics.CppMatcher(keypoint_metrics.CppMatcherConfig)
            matched_only_pairs, pem_pairs = [], []
            for frame_id in np.unique(gt_frameid):
                pred_pose = keypoint_data.PoseEstimationTensors(
                    keypoints=keypoint_data.KeypointsTensors(
                        location=tf.convert_to_tensor(np.asarray(pd_kp3d[pd_frameid == frame_id]).astype(np.float32), dtype=np.float32),
                        visibility=tf.convert_to_tensor(np.asarray(pd_kp3d_vis[pd_frameid == frame_id]).astype(np.float32), dtype=np.float32)
                    ),
                    box=keypoint_data.BoundingBoxTensors(
                        center=tf.convert_to_tensor(pd_boxes[pd_frameid == frame_id][..., :3], dtype=tf.float32),
                        size=tf.convert_to_tensor(pd_boxes[pd_frameid == frame_id][..., 3:6], dtype=tf.float32),
                        heading=tf.convert_to_tensor(pd_boxes[pd_frameid == frame_id][..., 6], dtype=tf.float32)
                    )
                )
                gt_pose = keypoint_data.PoseEstimationTensors(
                    keypoints=keypoint_data.KeypointsTensors(
                        location=tf.convert_to_tensor(np.asarray(gt_kp3d[gt_frameid == frame_id]).astype(np.float32), dtype=np.float32),
                        visibility=tf.convert_to_tensor(np.asarray(gt_kp3d_vis[gt_frameid == frame_id]).astype(np.float32), dtype=np.float32)
                    ),
                    box=keypoint_data.BoundingBoxTensors(
                        center=tf.convert_to_tensor(gt_boxes[gt_frameid == frame_id][..., :3], dtype=tf.float32),
                        size=tf.convert_to_tensor(gt_boxes[gt_frameid == frame_id][..., 3:6], dtype=tf.float32),
                        heading=tf.convert_to_tensor(gt_boxes[gt_frameid == frame_id][..., 6], dtype=tf.float32)
                    )
                )
                pair = keypoint_metrics.PoseEstimationPair(gt=gt_pose, pr=pred_pose)
                matched_only_pair = find_matched_only(matcher, pair)
                pem_pair = matcher.reorder(pair)
                pem_pairs.append(pem_pair)
                matched_only_pairs.append(matched_only_pair)

            # selected_pd_kp3d = tf.concat([sel.pd.keypoints.location for sel in matched_only_pairs], 0)
            # selected_pd_kp3d_vis = tf.concat([sel.pd.keypoints.visibility for sel in matched_only_pairs], 0)
            selected_pd_kp3d = keypoint_data.KeypointsTensors(
                location=tf.concat([sel.pr.keypoints.location for sel in matched_only_pairs], 0),
                visibility=tf.concat([sel.pr.keypoints.visibility for sel in matched_only_pairs], 0)
            )
            selected_pd_boxes = keypoint_data.BoundingBoxTensors(
                center=tf.concat([sel.pr.box.center for sel in matched_only_pairs], 0),
                size=tf.concat([sel.pr.box.size for sel in matched_only_pairs], 0),
                heading=tf.concat([sel.pr.box.heading for sel in matched_only_pairs], 0)
            )
            selected_gt_kp3d = keypoint_data.KeypointsTensors(
                location=tf.concat([sel.gt.keypoints.location for sel in matched_only_pairs], 0),
                visibility=tf.concat([sel.gt.keypoints.visibility for sel in matched_only_pairs], 0)
            )
            selected_gt_boxes = keypoint_data.BoundingBoxTensors(
                center=tf.concat([sel.gt.box.center for sel in matched_only_pairs], 0),
                size=tf.concat([sel.gt.box.size for sel in matched_only_pairs], 0),
                heading=tf.concat([sel.gt.box.heading for sel in matched_only_pairs], 0)
            )

            all_pd_kp3d = keypoint_data.KeypointsTensors(
                location=tf.concat([pem.pr.keypoints.location for pem in pem_pairs], 0),
                visibility=tf.concat([pem.pr.keypoints.visibility for pem in pem_pairs], 0)
            )
            all_pd_boxes = keypoint_data.BoundingBoxTensors(
                center=tf.concat([pem.pr.box.center for pem in pem_pairs], 0),
                size=tf.concat([pem.pr.box.size for pem in pem_pairs], 0),
                heading=tf.concat([pem.pr.box.heading for pem in pem_pairs], 0)
            )
            all_gt_kp3d = keypoint_data.KeypointsTensors(
                location=tf.concat([pem.gt.keypoints.location for pem in pem_pairs], 0),
                visibility=tf.concat([pem.gt.keypoints.visibility for pem in pem_pairs], 0)
            )
            all_gt_boxes = keypoint_data.BoundingBoxTensors(
                center=tf.concat([pem.gt.box.center for pem in pem_pairs], 0),
                size=tf.concat([pem.gt.box.size for pem in pem_pairs], 0),
                heading=tf.concat([pem.gt.box.heading for pem in pem_pairs], 0)
            )
            return (
                selected_pd_kp3d, selected_pd_boxes, selected_gt_kp3d, selected_gt_boxes,
                all_pd_kp3d, all_pd_boxes, all_gt_kp3d, all_gt_boxes
            )

        # NOTE: the computation from _alternative_matcher is a bit off from the _offical_matcher.
        # Just to in case if some some machine does not support the lastest version of the waymo package.
        (
            pred_kps, selected_pd_boxes, gt_kps, gt_boxTensor,
            all_pd_kp3d, all_pd_boxes, all_gt_kp3d, all_gt_boxes
        ) = _offical_matcher()  # _offical_matcher or _alternative_matcher

        tf.compat.v1.enable_eager_execution()

        all_metrics = keypoint_metrics.create_combined_metric(
            keypoint_metrics.DEFAULT_CONFIG_LASER)
        all_metrics.update_state((gt_kps, pred_kps, gt_boxTensor))
        result = all_metrics.result()

        try:
            pem = create_combined_metric(keypoint_metrics.DEFAULT_CONFIG_LASER)
            pem.update_state((all_gt_kp3d, all_pd_kp3d, all_gt_boxes))
            result.update(pem.result())
        except Exception as e:
            print("ALERT: PEM metric is not supported.")

        return result


def renew_box_info(gt_info):
    # Override gt_boxes_lidar with keypoints only boxes
    annos = gt_info['annos']
    annos = common_utils.drop_info_with_name(annos, name='unknown', ignore_name="keypoint")
    global_speed = np.pad(annos[
        'keypoint_box_speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
    speed = np.dot(global_speed, np.linalg.inv(gt_info['pose'][:3, :3].T))
    speed = speed[:, :2]
    gt_boxes_lidar = np.concatenate([
        annos['keypoint_box_center'], annos['keypoint_box_dimensions'],
        annos['keypoint_box_heading_angles'][..., np.newaxis], speed],
        axis=1
    )
    annos["gt_boxes_lidar"] = gt_boxes_lidar
    gt_info["annos"] = annos
    return gt_info


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Human'], help='')
    # 5 for train, 1 for test
    parser.add_argument('--sampled_interval', type=int, default=1, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results...')
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        cur_info = renew_box_info(cur_info)
        gt_infos_dst.append(cur_info)

    waymo_AP = eval.waymo_evaluation(
        pred_infos, gt_infos_dst, class_name=args.class_names, distance_thresh=1000, fake_gt_infos=False, has_kp=True
    )

    print(waymo_AP)


if __name__ == '__main__':
    main()
