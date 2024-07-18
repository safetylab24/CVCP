import numpy as np


def load_annotations(info: dict):
    result = {}
    if 'gt_boxes' in info and info['gt_boxes'] is not None:
        gt_boxes = np.array(info['gt_boxes']).astype(np.float32)
        gt_boxes[np.isnan(gt_boxes)] = 0
        result['annotations'] = {
            'boxes': gt_boxes,
            'names': np.array(info['gt_names']),
            'tokens': np.array(info['gt_boxes_token']),
            'velocities': np.array(info['gt_boxes_velocity']).astype(np.float32),
        }
    else:
        if not 'gt_boxes' in info:
            raise ValueError('No gt_boxes in info')
        elif info['gt_boxes'] is None:
            raise ValueError('gt_boxes is None')

    return result
