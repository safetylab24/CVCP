import json
import torch
import itertools
from pathlib import Path

from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np

from torchvision.transforms import ToTensor


def get_data(
    dataset_dir,
    cvt_metadata_path,
    bbox_label_path,
    split,
    version,
    tasks,
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    cvt_metadata_path = Path(cvt_metadata_path)
    bbox_label_path = Path(bbox_label_path)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split

    # gets a list of scene names from splits/nuscenes/[train/val].txt
    split_scenes = get_split(split)

    return [NuScenesGeneratedDataset(scene_name, cvt_metadata_path, bbox_label_path, tasks) for scene_name in split_scenes]

def parse_inputs(inputs: dict, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
    images = list()
    intrinsics = list()
    
    for image_path, I_original in zip(inputs['images'], inputs['intrinsics']):
        h_resize = h + top_crop
        w_resize = w

        image = Image.open(dataset_dir / image_path)

        image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
        image_new = image_new.crop(
            (0, top_crop, image_new.width, image_new.height))

        I = np.float32(I_original)
        I[0, 0] *= w_resize / image.width
        I[0, 2] *= w_resize / image.width
        I[1, 1] *= h_resize / image.height
        I[1, 2] *= h_resize / image.height
        I[1, 2] -= top_crop

        images.append(img_transform(image_new))
        intrinsics.append(torch.tensor(I))
    
    return images, intrinsics

def parse_labels(labels: dict, tasks) -> dict:
    
    # LoadPointCloudAnnotation transform
    gt_boxes = np.array(labels["gt_boxes"]).astype(np.float32)
    gt_boxes[np.isnan(gt_boxes)] = 0
    annotations = {
        "boxes": gt_boxes,
        "names": labels["gt_names"],
        "tokens": labels["gt_boxes_token"],
        "velocities": np.array(labels["gt_boxes_velocity"]).astype(np.float32),
    }
    
    # TODO: fix this first, then work on AssignLabel
    # Preprocess transform
    # gt_classes = np.array(
    #     [list(itertools.chain(*[t["class_names"] for t in tasks])).index(n) + 1 for n in labels["gt_names"]],
    #     dtype=np.int32,
    # )
    # gt_classes = list(itertools.chain(*[t["class_names"] for t in tasks]))
    
    #AssignLabel transform    
    max_objs = 500
    class_names_by_task = [t['class_names'] for t in tasks]

    pc_range = np.array((-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), dtype=np.float32)
    voxel_size = np.array((0.2, 0.2, 8), dtype=np.float32)
    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    feature_map_size = grid_size[:2]

    # reorganize the gt_dict by tasks
    task_masks = []
    flag = 0
    for class_name in class_names_by_task:
        task_masks.append(
            [
                np.where(
                    annotations["gt_classes"] == class_name.index(
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
            task_box.append(annotations["gt_boxes"][m])
            task_class.append(annotations["gt_classes"][m] - flag2)
            task_name.append(annotations["gt_names"][m])
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
    annotations["gt_classes"] = task_classes
    annotations["gt_names"] = task_names
    annotations["gt_boxes"] = task_boxes

    # draw_gaussian = draw_umich_gaussian

    hms, anno_boxes, inds, masks, cats = [], [], [], [], [] # labels

    for idx in range(len(tasks)):
        hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                        dtype=np.float32)

        # [reg, hei, dim, vx, vy, rots, rotc]
        anno_box = np.zeros((max_objs, 10), dtype=np.float32)

        ind = np.zeros((max_objs), dtype=np.int64)
        mask = np.zeros((max_objs), dtype=np.uint8)
        cat = np.zeros((max_objs), dtype=np.int64)

        num_objs = min(annotations['gt_boxes'][idx].shape[0], max_objs)

        for k in range(num_objs):
            cls_id = annotations['gt_classes'][idx][k] - 1

            w, l, h = annotations['gt_boxes'][idx][k][3], annotations['gt_boxes'][idx][k][4], annotations['gt_boxes'][idx][k][5]
            w, l = w / \
                voxel_size[0], l / \
                voxel_size[1]
            if w > 0 and l > 0:
                # radius = gaussian_radius(
                #     (l, w), min_overlap=self.gaussian_overlap)
                # radius = max(2, int(radius))

                # be really careful for the coordinate system of your box annotation.
                x, y, z = annotations['gt_boxes'][idx][k][0], annotations['gt_boxes'][idx][k][1], annotations['gt_boxes'][idx][k][2]

                coor_x, coor_y = (x - pc_range[0]) / voxel_size[0], \
                                    (y - pc_range[1]) / \
                    voxel_size[1]
                ct = np.array(
                    [coor_x, coor_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                    continue

                # draw_gaussian(hm[cls_id], ct, radius)

                new_idx = k
                x, y = ct_int[0], ct_int[1]

                cat[new_idx] = cls_id
                ind[new_idx] = y * feature_map_size[0] + x
                mask[new_idx] = 1
                
                vx, vy = annotations['gt_boxes'][idx][k][6:8]
                rot = annotations['gt_boxes'][idx][k][8]
                anno_box[new_idx] = np.concatenate(
                    (ct - (x, y), z, np.log(annotations['gt_boxes'][idx][k][3:6]),
                        np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)

        hms.append(hm)
        anno_boxes.append(anno_box)
        masks.append(mask)
        inds.append(ind)
        cats.append(cat)

    return {'hm': hms, 'anno_box': anno_boxes, 'ind': inds, 'mask': masks, 'cat': cats}

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def parse_data(inputs: dict, labels: dict, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
    images, intrinsics = parse_inputs(inputs, dataset_dir, h, w, top_crop, img_transform)
    labels = parse_labels(labels)

    return {
        'images': torch.stack(images, 0),
        'intrinsics': torch.stack(intrinsics, 0),
        'extrinsics': torch.tensor(np.float32(inputs['extrinsics'])),
        'labels': labels # fix this
    }


def get_split(version):
    if version == 'train':
        return (Path(__file__).parents[0] / 'splits/train.txt').read_text().splitlines()
    elif version == 'val':
        return (Path(__file__).parents[0] / 'splits/val.txt').read_text().splitlines()


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    ------ONE SCENE, MULTIPLE FRAMES-----
    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """

    def __init__(self, scene_name, cvt_metadata_path, bbox_label_path, tasks):
        self.samples = json.loads((Path(cvt_metadata_path) / f'{scene_name}.json').read_text())
        self.labels = json.loads((Path(bbox_label_path) / f'{scene_name}.json').read_text())
        self.tasks = tasks
        
        assert len(self.samples) == len(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # represents individual frame data from JSON as a dictionary
        inputs = self.samples[idx]
        labels = self.labels[idx]
                
        images, intrinsics = parse_inputs(
            inputs, 
            dataset_dir=Path('/home/vrb230004/media/datasets/nuscenes'), 
            h=224, w=480, top_crop=24, 
            img_transform=ToTensor())
        
        labels = parse_labels(labels, self.tasks)

        return {
            'images': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(inputs['extrinsics'])),
            'labels': labels # fix this
        }

def get_train_dataloader(dataset_dir=None, cvt_metadata_path=None, bbox_label_path=None, version='v1.0-trainval', batch_size=1, num_workers=1, num_classes=2, tasks=None):
    datasets = get_data(
        dataset_dir=dataset_dir,
        cvt_metadata_path=cvt_metadata_path,
        bbox_label_path=bbox_label_path,
        split='train',
        version=version,
        num_classes=num_classes,
        tasks=tasks
    )
    data = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False
    )

    return dataloader


def get_val_dataloader(dataset_dir=None, cvt_metadata_path=None, bbox_label_path=None, version='v1.0-trainval', batch_size=1, num_workers=1, num_classes=2, tasks=None):
    datasets = get_data(
        dataset_dir=dataset_dir,
        cvt_metadata_path=cvt_metadata_path,
        bbox_label_path=bbox_label_path,
        split='val',
        version=version,
        num_classes=num_classes,
        tasks=tasks
    )
    data = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False
    )

    return dataloader
