# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os

from torch.utils.data import DataLoader, Subset

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils

FUSION_DATASET_MAP = {
    'early': 'EarlyFusionDataset',
    'late': 'LateFusionDataset',
    'intermediate': 'IntermediateFusionDataset',
}


def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--hypes_yaml', type=str, default='',
                        help='optional visualization yaml path.')
    parser.add_argument('--fusion_method', type=str, default='early',
                        choices=sorted(FUSION_DATASET_MAP.keys()),
                        help='fusion strategy for data-layer visualization.')
    parser.add_argument('--color_mode', type=str, default="intensity",
                        choices=['constant', 'intensity', 'z-value'],
                        help='lidar color rendering mode.')
    parser.add_argument('--frame_index', type=int, default=None,
                        help='starting frame index. Negative values count '
                             'from the end, e.g. -1 for the last frame.')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='number of frames to process. Use 0 or a '
                             'negative value for continuous playback, or for '
                             'exporting until the dataset end.')
    parser.add_argument('--save_path', type=str, default='',
                        help='single output image path.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='directory for batch image export.')
    parser.add_argument('--width', type=int, default=1920,
                        help='output width in pixels.')
    parser.add_argument('--height', type=int, default=1080,
                        help='output height in pixels.')
    parser.add_argument('--point_size', type=float, default=1.0,
                        help='Open3D point size.')
    parser.add_argument('--background', type=str, default='dark',
                        choices=sorted(vis_utils.BACKGROUND_PRESETS.keys()),
                        help='background preset.')
    parser.add_argument('--hide_boxes', action='store_true',
                        help='hide ground-truth boxes in the output image.')
    parser.add_argument('--headless', action='store_true',
                        help='use matplotlib fallback for saving images.')
    return parser.parse_args()


def load_visualization_hypes(opt):
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes_yaml = opt.hypes_yaml or \
        os.path.join(current_path, '../hypes_yaml/visualization.yaml')
    params = load_yaml(hypes_yaml)
    params['fusion']['core_method'] = FUSION_DATASET_MAP[opt.fusion_method]
    return params, current_path


def build_output_paths(current_path, opt, frame_indices):
    if opt.save_path and len(frame_indices) > 1:
        raise ValueError('--save_path only supports exporting a single frame. '
                         'Use --save_dir for batch export.')

    if opt.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(opt.save_path)),
                    exist_ok=True)
        return [opt.save_path]

    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)
        output_dir = opt.save_dir
    else:
        output_dir = os.path.join(current_path, '../logs',
                                  '%s_%s_export' % (opt.fusion_method,
                                                    opt.color_mode))
        os.makedirs(output_dir, exist_ok=True)

    return [os.path.join(output_dir,
                         '%s_%s_frame_%05d.png' %
                         (opt.fusion_method, opt.color_mode, frame_idx))
            for frame_idx in frame_indices]


def get_collated_sample(dataset, frame_idx):
    return dataset.collate_batch_test([dataset[frame_idx]])


def export_frames(dataset, params, current_path, opt):
    frame_indices = vis_utils.resolve_frame_indices(len(dataset),
                                                    opt.frame_index,
                                                    opt.num_frames)
    output_paths = build_output_paths(current_path, opt, frame_indices)

    for frame_idx, output_path in zip(frame_indices, output_paths):
        sample_batched = get_collated_sample(dataset, frame_idx)
        try:
            if opt.headless:
                raise AttributeError('background_color headless fallback')

            vis_utils.visualize_single_sample_dataloader(
                sample_batched['ego'],
                None,
                params['postprocess']['order'],
                mode=opt.color_mode,
                save_path=output_path,
                width=opt.width,
                height=opt.height,
                point_size=opt.point_size,
                background_color=opt.background,
                include_boxes=not opt.hide_boxes)
        except AttributeError as exc:
            if not opt.headless and "background_color" not in str(exc):
                raise
            vis_utils.save_sequence_sample_plt(
                sample_batched['ego'],
                params['postprocess']['order'],
                params['preprocess']['cav_lidar_range'],
                output_path,
                width=opt.width,
                height=opt.height,
                mode=opt.color_mode,
                point_size=opt.point_size,
                background_color=opt.background,
                include_boxes=not opt.hide_boxes)
        print('saved frame %d to %s' % (frame_idx, output_path))


def playback_frames(dataset, params, opt):
    if opt.frame_index is None and opt.num_frames <= 0:
        playback_dataset = dataset
        max_frames = None
    else:
        frame_indices = vis_utils.resolve_frame_indices(len(dataset),
                                                        opt.frame_index,
                                                        opt.num_frames)
        playback_dataset = Subset(dataset, frame_indices)
        max_frames = None if opt.num_frames <= 0 else len(frame_indices)

    data_loader = DataLoader(playback_dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)

    vis_utils.visualize_sequence_dataloader(
        data_loader,
        params['postprocess']['order'],
        color_mode=opt.color_mode,
        max_frames=max_frames,
        width=opt.width,
        height=opt.height,
        point_size=opt.point_size,
        background_color=opt.background,
        include_boxes=not opt.hide_boxes)


if __name__ == '__main__':
    opt = vis_parser()
    params, current_path = load_visualization_hypes(opt)
    opencood_dataset = build_dataset(params, visualize=True, train=False)

    if opt.save_path or opt.save_dir:
        export_frames(opencood_dataset, params, current_path, opt)
    else:
        playback_frames(opencood_dataset, params, opt)
