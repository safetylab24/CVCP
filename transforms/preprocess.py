import numpy as np

from models.centerpoint.utils.box_np_ops import limit_period
import models.centerpoint.utils.preprocess as prep
import itertools

from models.centerpoint.utils.center_utils import draw_umich_gaussian, gaussian_radius


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def preprocess(info, tasks,
               global_rotation_noise=[-0.3925, 0.3925],
               global_scale_noise=[0.95, 1.05],
               global_translate_std=0):

    class_names = list(itertools.chain(*[t['class_names'] for t in tasks]))

    anno_dict = info['annotations']

    gt_dict = {
        'gt_boxes': anno_dict['boxes'],
        'gt_names': np.array(anno_dict['names']).reshape(-1),
    }

    selected = drop_arrays_by_name(
        gt_dict['gt_names'], ['DontCare', 'ignore', 'UNKNOWN']
    )

    _dict_select(gt_dict, selected)

    gt_boxes_mask = np.array(
        [n in class_names for n in gt_dict['gt_names']], dtype=np.bool_
    )

    _dict_select(gt_dict, gt_boxes_mask)

    gt_classes = np.array(
        [class_names.index(n) + 1 for n in gt_dict['gt_names']],
        dtype=np.int32,
    )
    gt_dict['gt_classes'] = gt_classes

    gt_dict['gt_boxes'] = prep.random_flip_both(
        gt_dict['gt_boxes'])

    gt_dict['gt_boxes'] = prep.global_rotation(
        gt_dict['gt_boxes'], rotation=global_rotation_noise
    )
    gt_dict['gt_boxes'] = prep.global_scaling_v2(
        gt_dict['gt_boxes'], *global_scale_noise
    )
    gt_dict['gt_boxes'] = prep.global_translate_(
        gt_dict['gt_boxes'], noise_translate_std=global_translate_std
    )
    
    info['annotations'] = gt_dict

    return info


def flatten(box):
    return np.concatenate(box, axis=0)


'''Return CenterNet training labels like heatmap, height, offset'''


def assign_label(result, tasks, gaussian_overlap=0.1, max_objs=500, min_radius=2, out_size_factor=4, voxel_size=(0.2, 0.2, 8),
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
    """
    Assigns labels to the given result based on the specified tasks.

    Args:
        result (dict): The result dictionary containing annotations.
        tasks (list): A list of dictionaries representing the tasks.
        gaussian_overlap (float, optional): The minimum overlap required for Gaussian radius calculation. Defaults to 0.1.
        max_objs (int, optional): The maximum number of objects. Defaults to 500.
        min_radius (int, optional): The minimum radius for Gaussian radius calculation. Defaults to 2.
        out_size_factor (int, optional): The output size factor. Defaults to 4.
        voxel_size (tuple, optional): The voxel size. Defaults to (0.2, 0.2, 8).
        pc_range (tuple, optional): The point cloud range. Defaults to (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0).

    Returns:
        dict: The labels dictionary containing heatmaps, annotation boxes, indices, masks, categories, and number of objects.
    """
    class_names_by_task = [t['class_names'] for t in tasks]
    num_classes_by_task = [t['num_class'] for t in tasks]

    labels = {}

    # Calculate output featuremap size
    pc_range = np.array(pc_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)

    feature_map_size = grid_size[:2] // out_size_factor

    gt_dict = result['annotations']

    # reorganize the gt_dict by tasks
    task_masks = []
    flag = 0
    for class_name in class_names_by_task:
        task_masks.append(
            [
                np.where(
                    gt_dict['gt_classes'] == class_name.index(
                        i) + 1 + flag
                )
                for i in class_name
            ]
        )
        flag += len(class_name)

    task_boxes = []
    task_classes = []
    task_names = []
    flag2 = 0
    for idx, mask in enumerate(task_masks):
        task_box = []
        task_class = []
        task_name = []
        for m in mask:
            task_box.append(gt_dict['gt_boxes'][m])
            task_class.append(gt_dict['gt_classes'][m] - flag2)
            task_name.append(gt_dict['gt_names'][m])
        task_boxes.append(np.concatenate(task_box, axis=0))
        task_classes.append(np.concatenate(task_class))
        task_names.append(np.concatenate(task_name))
        flag2 += len(mask)

    for task_box in task_boxes:
        # limit rad to [-pi, pi]
        task_box[:, -1] = limit_period(
            task_box[:, -1], offset=0.5, period=np.pi * 2
        )

    # print(gt_dict.keys())
    gt_dict['gt_classes'] = task_classes
    gt_dict['gt_names'] = task_names
    gt_dict['gt_boxes'] = task_boxes

    result['annotations'] = gt_dict

    draw_gaussian = draw_umich_gaussian

    hms, anno_boxs, inds, masks, cats = [], [], [], [], []

    for idx, task in enumerate(tasks):
        hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                        dtype=np.float32)

        # [reg, hei, dim, vx, vy, rots, rotc]
        anno_box = np.zeros((max_objs, 10), dtype=np.float32)

        ind = np.zeros((max_objs), dtype=np.int64)
        mask = np.zeros((max_objs), dtype=np.uint8)
        cat = np.zeros((max_objs), dtype=np.int64)

        num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)

        for k in range(num_objs):
            cls_id = gt_dict['gt_classes'][idx][k] - 1

            w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], gt_dict['gt_boxes'][idx][k][5]
            w, l = w / voxel_size[0] / out_size_factor, l / \
                voxel_size[1] / out_size_factor
            if w > 0 and l > 0:
                radius = gaussian_radius(
                    (l, w), min_overlap=gaussian_overlap)
                radius = max(min_radius, int(radius))

                # be really careful for the coordinate system of your box annotation.
                x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], gt_dict['gt_boxes'][idx][k][2]

                coor_x, coor_y = (
                    x - pc_range[0]) / voxel_size[0] / out_size_factor, (y - pc_range[1]) / voxel_size[1] / out_size_factor

                ct = np.array(
                    [coor_x, coor_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                    continue

                draw_gaussian(hm[cls_id], ct, radius)

                new_idx = k
                x, y = ct_int[0], ct_int[1]

                cat[new_idx] = cls_id
                ind[new_idx] = y * feature_map_size[0] + x
                mask[new_idx] = 1

                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                rot = gt_dict['gt_boxes'][idx][k][8]
                anno_box[new_idx] = np.concatenate(
                    (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                        np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)

        hms.append(hm)
        anno_boxs.append(anno_box)
        masks.append(mask)
        inds.append(ind)
        cats.append(cat)

    labels.update({'hm': hms, 'anno_box': anno_boxs,
                    'ind': inds, 'mask': masks, 'cat': cats, 'num_objs': num_objs})

    return labels
