import torchvision
import pathlib
from PIL import Image
import torch
import numpy as np


class LoadDataTransform(torchvision.transforms.ToTensor):
    """
    Custom data transformation class for loading and preprocessing data.

    Args:
        dataset_dir (str): Directory path of the dataset.
        labels_dir (str): Directory path of the labels.
        image_config (dict): Configuration parameters for image processing.
        num_classes (int): Number of classes in the dataset.
        augment (str, optional): Type of augmentation to apply. Defaults to 'none'.

    Attributes:
        dataset_dir (Path): Path object representing the dataset directory.
        labels_dir (Path): Path object representing the labels directory.
        image_config (dict): Configuration parameters for image processing.
        num_classes (int): Number of classes in the dataset.
        img_transform (ToTensor): Image transformation object for converting images to tensors.
        to_tensor (callable): Callable object for converting data to tensors.

    Methods:
        get_cameras: Crops, resizes, and augments images.
        decode: Decodes the image.
        get_bev: Retrieves the bird's eye view (BEV) image.
        __call__: Applies the data transformation.

    """

    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes

        self.img_transform = torchvision.transforms.ToTensor()
        self.to_tensor = super().__call__

    # Rest of the code...
class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes

        self.img_transform = torchvision.transforms.ToTensor()
        self.to_tensor = super().__call__

    # crops, resizes, and augments images
    # returns dict of tensors for camids, images, intrinsics, extrinsics
    def get_cameras(self, sample, h, w, top_crop):
        '''
        Note: we invert I and E here for convenience.
        '''
        images = list()
        intrinsics = list()

        for image_path, I_original in zip(sample.images, sample.intrinsics):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize(
                (w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop(
                (0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
        }

    def decode(self, img, n):
        '''
        returns (h, w, n) np.int32 {0, 1}
        '''
        # [[[0, 1, ... 11]]] - (1, 1, 12)
        shift = np.arange(n, dtype=np.int32)[None, None]

        # (200, 200, 1), [..., None] adds a new axis at the end (instead of the beginning)
        x = np.array(img)[..., None]
        x = (x >> shift) & 1

        return x

    def get_bev(self, sample):
        scene_dir = self.labels_dir / sample.scene
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)

            shift = np.arange(self.num_classes, dtype=np.int32)[None, None]

            # (200, 200, 1), [..., None] adds a new axis at the end (instead of the beginning)
            x = np.array(bev)[..., None]
            bev = (x >> shift) & 1

            # bev = decode(bev, self.num_classes) # converts to binary numpy array
            # converts to uint8 - 0 or 255 values (now visible)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)

        result = {
            'bev': bev,  # just has road features
            'view': torch.tensor(sample.view),
        }

        if 'visibility' in sample:  # visibility mask - vehicles
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:  # center of vehicles
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])

        if 'pose' in sample:  # 4x4 pose matrix
            result['pose'] = np.float32(sample['pose'])

        return result

    def __call__(self, frame):
        # batch = 1 frame of data as a dict

        result = dict()
        # adds cam_ids for all 6 cameras, image stack, intrinsics matrix, extrinsics-to-result matrix
        result.update(self.get_cameras(frame, **self.image_config))
        # adds bev, view matrix, visibility mask, center mask, pose-to-result matrix
        result.update(self.get_bev(frame))

        return result
