import argparse
import time
from datetime import datetime
from pathlib import Path

from visual_utils import open3d_vis_utils as V

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import WaymoDatasetKP
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='../checkpoints/voxelkp_checkpoint.pth')
    parser.add_argument('--data_path', type=str, default='/home/shij0c/git/KP3D/OpenPCDet/data/waymo', help='specify the point cloud data file or directory')
    parser.add_argument('--gt_only', action="store_true", help="show results with ground truth annotation.")
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    cfg.DATA_CONFIG.REMOVE_BOXES_WITHOUT_KEYPOINTS = False
    # cfg.MODEL.DENSE_HEAD.IOU_BRANCH = False
    cfg.DATA_CONFIG.LABEL_MAPPING = None

    demo_dataset = WaymoDatasetKP(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,  # For visualizing augmentations
        root_path=Path(args.data_path), logger=logger, inference_mode=False
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    time_used = []
    count = 0

    blocking = True
    with torch.inference_mode():
        for idx, data_dict in enumerate(demo_dataset):

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)

            start = datetime.now()
            pred_dicts, _ = model.forward(data_dict)
            time_used.append((datetime.now() - start).total_seconds())

            count+= 1
            print(f"Time used for frame {count}:", time_used[-1], "Running Average (frames/s):", 1 / (sum(time_used) / count))

            if count == 1 or blocking:
                vis = V.draw_scenes(
                    # points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    # ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    points=data_dict['points'][:, 1:],
                    # gt_boxes=data_dict['gt_boxes'][0],
                    # gt_keypoints=data_dict['keypoint_location'][0],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_keypoints=pred_dicts[0]['pred_kps'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    points_to_keep_boxes=pred_dicts[0]['pred_boxes'] if args.gt_only else None,
                    blocking=blocking
                )

                vis.run()

                vis.poll_events()
                vis.update_renderer()

                time.sleep(1)

                vis.destroy_window()
                del vis
            else:
                # Buggy
                vis = V.update_scenes(
                    vis=vis,
                    points=data_dict['points'][:, 1:],
                    # gt_boxes=data_dict['gt_boxes'][0],
                    # gt_keypoints=data_dict['keypoint_location'][0],
                    # ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_keypoints=pred_dicts[0]['pred_kps'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    points_to_keep_boxes=pred_dicts[0]['pred_boxes'] if args.gt_only else None,
                )
                time.sleep(1)

        if not blocking:
            vis.destroy_window()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
