from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from .loss_utils import _transpose_and_gather_feat


class OksLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        code_weight (float): Weight for the loss.
    """

    def __init__(self, code_weights: list = None):
        super().__init__()

        self.code_weights = np.array(code_weights, dtype=np.float32)
        self.code_weights = torch.from_numpy(self.code_weights).reshape(-1, 3).mean(-1)[None]

    def forward(self,
                output: Tensor,
                target: Tensor,
                mask: Tensor,
                bboxes: Optional[Tensor] = None) -> Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 3, N
                is the number of anchors, k is the number of keypoints,
                and 3 are the xyz coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            mask (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 6,
                where 6 are the xyz dx dy dz coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """
        oks = self.compute_oks(output, target, mask,bboxes)
        loss = 1 - oks
        return loss

    def compute_oks(self,
                    output: Tensor,
                    target: Tensor,
                    mask: Tensor,
                    bboxes: Optional[Tensor] = None) -> Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 3, where N
                is batch size, k is the number of keypoints, and 3 are the
                xyz coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            mask (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 6,
                where 6 are the xyz dx dy dz coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """

        dist = torch.norm(output - target, dim=-1)

        if bboxes is not None:
            # area = torch.norm(bboxes[..., 3:6] - bboxes[..., :3], dim=-1)
            area = torch.norm(bboxes[..., 3:6], dim=-1)
            dist = dist / area.clip(min=1e-8).unsqueeze(-1)

        return (torch.exp(-dist.pow(2) / 2) * torch.as_tensor(
            self.code_weights, device=dist.device) * mask).sum(
            dim=-1) / mask.sum(dim=-1).clip(min=1e-8)


def _reg_loss(regr, gt_regr, visibility, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask
    vis_mask = (visibility > 0).float()

    loss = torch.abs(regr - gt_regr) * vis_mask
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _reg_loss_smooth(regr, gt_regr, visibility, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask
    vis_mask = (visibility > 0).float()

    loss = F.smooth_l1_loss(regr, gt_regr, reduction='none', beta=0.1) * vis_mask
    # loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


class RegLossSparseKP(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossSparseKP, self).__init__()

    def forward(self, output, mask, visibility, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim)
            mask: (batch x max_objects)
            visibility: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """

        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        loss = _reg_loss_smooth(pred, target, visibility, mask)
        return loss


class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        bone_joints_order (list): Indices of each joint's parent joint.
            Joints not in the list will be excluded.
        joints_order (list):Indices of the order of each joint.
            Joints not in the list will be excluded.
    """

    def __init__(self, bone_joints_order: List[int], joints_order: List[int] = None):
        super().__init__()
        self.bone_joints_order = bone_joints_order
        self.joints_order = joints_order
        # self._huber = nn.HuberLoss()

        if joints_order is not None and len(joints_order) != len(bone_joints_order):
            raise ValueError

    def get_bone_size(self, output, target, visibility=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        assert len(output.shape) == 3 and len(target.shape) == 3, (output.shape, target.shape)
        if self.joints_order is None:
            joints_order = range(output.size(1))
        else:
            joints_order = self.joints_order

        # Get mask for valid pairs where both x and y are valid
        if visibility is not None:
            with torch.no_grad():
                mask = ((visibility[:, joints_order] > 0) & (visibility[:, self.bone_joints_order] > 0)).float()
        else:
            mask = 1.

        output_bone = torch.norm(
            output[:, joints_order] - output[:, self.bone_joints_order], dim=-1) * mask
        target_bone = torch.norm(
            target[:, joints_order] - target[:, self.bone_joints_order], dim=-1) * mask

        return output_bone, target_bone

    def forward(self, output, target, visibility=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        output_bone, target_bone = self.get_bone_size(output, target, visibility)

        loss = torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0))

        return loss


class BoneLossCenterNet(nn.Module):
    """BoneLoss for centerNet outputs.
    """

    def __init__(self, bone_joints_order: List[int], joints_order: List[int] = None):
        super(BoneLossCenterNet, self).__init__()
        self.bone_loss = BoneLoss(bone_joints_order, joints_order)

    def forward(self, output, mask, ind=None, target=None, batch_index=None, visibility=None):
        dim_size = output.size(1)
        output = output.view(output.size(0), -1)
        target = target.view(target.size(0), target.size(1), -1)
        if visibility is not None:
            visibility = visibility.view(visibility.size(0), visibility.size(1), -1)

        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        mask_kp = mask.unsqueeze(2).expand_as(target).float()
        isnotnan = (~ torch.isnan(target)).float()
        mask_kp = mask_kp * isnotnan
        pred = pred * mask_kp
        target = target * mask_kp

        if visibility is not None:
            mask_vis = mask.unsqueeze(2).expand_as(visibility).float()
            isnotnan = (~ torch.isnan(visibility)).float()
            mask_vis = mask_vis * isnotnan
            visibility = visibility * mask_vis

        B = pred.size(0)
        pred = pred.view(B * pred.size(1), dim_size, -1)
        target = target.view(B * target.size(1), dim_size, -1)

        if visibility is not None:
            visibility = visibility.view(B * visibility.size(1), dim_size)
        # return pred, target, visibility
        loss = self.bone_loss(pred, target, visibility)
        return loss
        
    # def forward(self, output, mask, ind=None, target=None, batch_index=None, visibility=None):
    #     """
    #     Args:
    #         output: (N x dim x 3)
    #         mask: (batch x max_objects)
    #         ind: (batch x max_objects)
    #         target: (batch x max_objects x dim x 3)
    #     Returns:
    #     """
    #     dim_size = output.size(1)
    #     output = output.view(output.size(0), -1)
    #     target = target.view(target.size(0), target.size(1), -1)
    #     pred = []
    #     batch_size = mask.shape[0]
    #     for bs_idx in range(batch_size):
    #         batch_inds = batch_index==bs_idx
    #         pred.append(output[batch_inds][ind[bs_idx]])
    #     pred = torch.stack(pred)
    #     # assert False, (pred.shape, output.shape, target.shape)

    #     mask = mask.unsqueeze(2).expand_as(target).float()
    #     isnotnan = (~ torch.isnan(target)).float()
    #     mask *= isnotnan
    #     pred = pred * mask
    #     target = target * mask

    #     B = pred.size(0)
    #     pred = pred.view(B * pred.size(1), dim_size, -1)
    #     target = target.view(B * target.size(1), dim_size, -1)

    #     loss = self.bone_loss(pred, target, visibility)
    #     return loss


class SkeletonLoss(nn.Module):

    def __init__(
        self, bone_joints_order: List[int], joints_order: List[int] = None,
        angle_end_joints_a_order: List[int] = None,
        angle_end_joints_b_order: List[int] = None,
        angle_mid_joints_order: List[int] = None,
    ):
        super(SkeletonLoss, self).__init__()
        self.bone_loss = BoneLoss(bone_joints_order, joints_order)
        self.angle_end_joints_a_order = angle_end_joints_a_order
        self.angle_end_joints_b_order = angle_end_joints_b_order
        self.angle_mid_joints_order = angle_mid_joints_order
        self._huber = nn.HuberLoss()

    def preproc(self, output, mask, visibility, ind=None, target=None, batch_index=None):
        dim_size = output.size(1)
        output = output.view(output.size(0), -1)
        target = target.view(target.size(0), target.size(1), -1)
        visibility = visibility.view(visibility.size(0), visibility.size(1), -1)
        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        mask_kp = mask.unsqueeze(2).expand_as(target).float()
        isnotnan = (~ torch.isnan(target)).float()
        mask_kp = mask_kp * isnotnan
        pred = pred * mask_kp
        target = target * mask_kp
        mask_vis = mask.unsqueeze(2).expand_as(visibility).float()
        isnotnan = (~ torch.isnan(visibility)).float()
        mask_vis = mask_vis * isnotnan
        visibility = visibility * mask_vis

        B = pred.size(0)
        pred = pred.view(B * pred.size(1), dim_size, -1)
        target = target.view(B * target.size(1), dim_size, -1)
        visibility = visibility.view(B * visibility.size(1), dim_size)
        return pred, target, visibility

    def _compute_angle_by_joints(self, end_joint_a, end_joint_b, mid_joint):
        eps = 1e-6
        end_joint_a_to_mid_joint = end_joint_a - mid_joint
        end_joint_b_to_mid_joint = end_joint_b - mid_joint

        angle = torch.acos(
            (end_joint_a_to_mid_joint * end_joint_b_to_mid_joint).sum(-1) /
            ((torch.norm(end_joint_a_to_mid_joint, dim=-1, p=2) + eps) * (torch.norm(end_joint_b_to_mid_joint, dim=-1, p=2) + eps))
        )
        return angle

    def _compute_yaw_pitch_roll_by_joints(self, end_joint_a, end_joint_b, mid_joint):
        end_joint_a_to_mid_joint = end_joint_a - mid_joint
        end_joint_b_to_mid_joint = end_joint_b - mid_joint

        # Direction vectors
        dir_a = torch.nn.functional.normalize(end_joint_a_to_mid_joint, dim=-1, p=2)
        dir_b = torch.nn.functional.normalize(end_joint_b_to_mid_joint, dim=-1, p=2)

        # Yaw
        yaw_a = torch.atan2(dir_a[:, :, 1], dir_a[:, :, 0] + 1e-7)
        yaw_b = torch.atan2(dir_b[:, :, 1], dir_b[:, :, 0] + 1e-7)
        yaw_diff = yaw_a - yaw_b

        # Pitch
        pitch_a = torch.asin(dir_a[:, :, 1])
        pitch_b = torch.asin(dir_b[:, :, 1])
        pitch_diff = pitch_a - pitch_b

        # Roll
        right_a = torch.cross(dir_a, torch.tensor([[[0., 0., 1.]]], device=end_joint_a.device))
        right_b = torch.cross(dir_b, torch.tensor([[[0., 0., 1.]]], device=end_joint_b.device))
        roll_a = torch.atan2(right_a[:, :, 1], right_a[:, :, 0] + 1e-7)
        roll_b = torch.atan2(right_b[:, :, 1], right_b[:, :, 0] + 1e-7)
        roll_diff = roll_a - roll_b

        return torch.stack([yaw_diff, pitch_diff, roll_diff], dim=-1)

    def get_angle_size(self, pred, target, visibility):

        # Get mask for valid pairs where both x and y are valid
        with torch.no_grad():
            mask = (
                (visibility[:, self.angle_end_joints_a_order] > 0) & (visibility[:, self.angle_end_joints_b_order] > 0) & (visibility[:, self.angle_mid_joints_order] > 0)
            ).float()

        target_angle = self._compute_angle_by_joints(
            target[:, self.angle_end_joints_a_order],
            target[:, self.angle_end_joints_b_order],
            target[:, self.angle_mid_joints_order],
        ) * mask
        pred_angle = self._compute_angle_by_joints(
            pred[:, self.angle_end_joints_a_order],
            pred[:, self.angle_end_joints_b_order],
            pred[:, self.angle_mid_joints_order],
        ) * mask
        return pred_angle, target_angle

    def get_yaw_pitch_roll_size(self, pred, target, visibility):

        # Get mask for valid pairs where both x and y are valid
        with torch.no_grad():
            mask = (
                (visibility[:, self.angle_end_joints_a_order] > 0) & (visibility[:, self.angle_end_joints_b_order] > 0) & (visibility[:, self.angle_mid_joints_order] > 0)
            ).float().unsqueeze(-1)
        
        tg_a, tg_b, tg_mid = target[:, self.angle_end_joints_a_order], target[:, self.angle_end_joints_b_order], target[:, self.angle_mid_joints_order]
        pd_a, pd_b, pd_mid = pred[:, self.angle_end_joints_a_order], pred[:, self.angle_end_joints_b_order], pred[:, self.angle_mid_joints_order]

        target_angle = self._compute_yaw_pitch_roll_by_joints(tg_a, tg_b, tg_mid) * mask
        pred_angle = self._compute_yaw_pitch_roll_by_joints(pd_a, pd_b, pd_mid) * mask
        return pred_angle, target_angle

    def forward(self, output, mask, visibility, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim x 3)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim x 3)
        Returns:
        """

        pred, target, vis = self.preproc(output, mask, visibility, ind, target, batch_index)
        output_bone, target_bone = self.bone_loss.get_bone_size(pred, target, vis)
        output_angle, target_angle = self.get_yaw_pitch_roll_size(pred, target, vis)

        return self._huber(output_bone, target_bone) + self._huber(output_angle, target_angle) * 0.1
