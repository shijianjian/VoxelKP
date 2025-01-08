"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import os
import open3d
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import matplotlib
import numpy as np

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

box_colormap = [
    [1, 1, 1],
    [.4, .2, .4],
    [0, 1, 1],
    [1, 1, 0],
]


def create_arrow(pos=[0, 0, 0], degree=0, color=[1, 0, 0]):
    length = 0.5
    radius = 0.05
    
    arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2, cylinder_height=length, cone_height=length/2)
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color) # Red color
    import math
    radian = math.radians(degree)
    rotation_axis = [0, math.radians(90), radian] # rotate around z-axis
    R = arrow.get_rotation_matrix_from_axis_angle(rotation_axis) 
    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(pos)
    
    return arrow


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba



def remove_background_points(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d[:, :7] + np.array([[0, 0, 0, 0.1, 0.1, 0.1, 0]]))
    points = points[~(point_masks.sum(axis=0) == 0)]

    return points


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes.

        Taken from https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def obtain_all_geometry(
    points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
    gt_keypoints=None, ref_keypoints=None, point_colors=None, draw_origin=False, points_to_keep_boxes=None,
):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.cpu().numpy()
    if isinstance(ref_keypoints, torch.Tensor):
        ref_keypoints = ref_keypoints.cpu().numpy()
    if isinstance(points_to_keep_boxes, torch.Tensor):
        points_to_keep_boxes = points_to_keep_boxes.cpu().numpy()

    if points_to_keep_boxes is not None:
        points = remove_background_points(points, points_to_keep_boxes)
        # N = ref_keypoints.shape[1]
        # ref_keypoints = remove_background_points(ref_keypoints.reshape(-1, 3), points_to_keep_boxes).reshape(-1, N, 3)[:points.shape[0]]

    all_geometry = []

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        z_colors = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
        pts.colors = o3d.utility.Vector3dVector(np.tile(z_colors[:, None], [1, 3]))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)


    all_geometry.append(pts)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        all_geometry.append(axis_pcd)

    if gt_boxes is not None:
        box_geo = draw_box(gt_boxes, (.8, .2, .8))
        all_geometry.extend(box_geo)

    if ref_boxes is not None:
        box_geo = draw_box(ref_boxes, (.4, .2, .4), ref_labels, ref_scores)
        all_geometry.extend(box_geo)

    if gt_keypoints is not None:
        kp_geo = draw_human(gt_keypoints, (1, .2, .5))
        all_geometry.extend(kp_geo)
    
    if ref_keypoints is not None:
        kp_geo = draw_human(ref_keypoints, (1., 0., .2))
        all_geometry.extend(kp_geo)

    return all_geometry


def draw_scenes(
    points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
    gt_keypoints=None, ref_keypoints=None, point_colors=None, draw_origin=False, points_to_keep_boxes=None,
    blocking=True
):
    all_geometry = obtain_all_geometry(
        points, gt_boxes=gt_boxes, ref_boxes=ref_boxes,
        ref_labels=ref_labels, ref_scores=ref_scores,
        gt_keypoints=gt_keypoints, ref_keypoints=ref_keypoints,
        point_colors=point_colors, draw_origin=draw_origin, points_to_keep_boxes=points_to_keep_boxes,
    )

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1061)

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    for geo in all_geometry:
        vis.add_geometry(geo)

    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(
        os.path.join(os.path.dirname(__file__), "./camera_pose_2.json"))
    ctr.convert_from_pinhole_camera_parameters(parameters)
    
    return vis


def update_scenes(
    vis, points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
    gt_keypoints=None, ref_keypoints=None, point_colors=None, draw_origin=False, points_to_keep_boxes=None,
):
    all_geometry = obtain_all_geometry(
        points, gt_boxes=gt_boxes, ref_boxes=ref_boxes,
        ref_labels=ref_labels, ref_scores=ref_scores,
        gt_keypoints=gt_keypoints, ref_keypoints=ref_keypoints,
        point_colors=point_colors, draw_origin=draw_origin, points_to_keep_boxes=points_to_keep_boxes,
    )

    vis.update_geometry(all_geometry[0])
    for geo in all_geometry[1:]:
        vis.add_geometry(geo)

    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(
        os.path.join(os.path.dirname(__file__), "./camera_pose.json"))
    ctr.convert_from_pinhole_camera_parameters(parameters)

    vis.poll_events()
    vis.update_renderer()

    return vis


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, angle=False):
    all_geo = []
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        all_geo.append(line_set)
        corners = box3d.get_box_points()
        if angle:
            all_geo.append(create_arrow(corners[5], gt_boxes[i][6] * 180 / 3.14159265358, color))
    return all_geo


def draw_human(keypoints, color=( .8, .1, .1)):
    all_geo = []

    for points in keypoints:
        mid_hip = (points[7] + points[8]) / 2
        mid_shoulder = (points[1] + points[2]) / 2
        points = np.concatenate([points, [mid_hip, mid_shoulder]], axis=0)

        lines = [
            [13, 14],  # spine
            [15, 1], [1, 3], [3, 5],  # upper, left
            [15, 2], [2, 4], [4, 6],  # upper, right
            [7, 9], [9, 11],  # bottom, left
            [8, 10], [10, 12],  # bottom, right
            [14, 7], [14, 8]  # spine to hip
        ]
        colors = [color for i in range(len(lines))]
        # line_set = open3d.geometry.LineSet()
        # line_set.points = open3d.utility.Vector3dVector(points)
        # line_set.lines = open3d.utility.Vector2iVector(lines)
        # line_set.colors = open3d.utility.Vector3dVector(colors)
        # vis.add_geometry(line_set)

        line_mesh = LineMesh(points, lines, colors, radius=0.02)
        # line_mesh.add_line(vis)

        for cylinder in line_mesh.cylinder_segments:
            all_geo.append(cylinder)

    return all_geo
