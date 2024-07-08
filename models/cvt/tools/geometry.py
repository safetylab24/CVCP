import cv2
import numpy as np
import torch
from nuscenes.utils.data_classes import Box
from PIL import Image
from pyquaternion import Quaternion
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def resize_and_crop_image(img, resize_dims, crop):
    img = img.resize(resize_dims, resample=Image.BILINEAR)
    img = img.crop(crop)
    return img


def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    updated_intrinsics = intrinsics.clone()

    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = np.array([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = np.array([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = np.array([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=np.int32)

    return bev_resolution, bev_start_position, bev_dimension