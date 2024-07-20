from nuscenes import NuScenes
from pathlib import Path
import pickle
from pyquaternion import Quaternion
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# generate pkl for each scene
# each pkl will have the following structure:
# [{
#     "scene": "scene_name",
#     "token": sample_token
#     "intrinsics": [I1, I2, I3, I4, I5, I6, I7, I8, I9],
#     "extrinsics": [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16],
#     "images": [image_path1, image_path2, image_path3, image_path4, image_path5, image_path6},
# }, ...]


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2),
                       vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)


def parse_pose(record, *args, **kwargs):
    return get_pose(record['rotation'], record['translation'], *args, **kwargs)


def parse_sample_record(sample_record, camera_rig, scene_name, nusc):
    lidar_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    egolidar = nusc.get('ego_pose', lidar_record['ego_pose_token'])

    world_from_egolidarflat = parse_pose(egolidar, flat=True)
    egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

    cam_channels = []
    images = []
    intrinsics = []
    extrinsics = []

    for cam_idx in camera_rig:
        cam_channel = CAMERAS[cam_idx]
        cam_token = sample_record['data'][cam_channel]

        cam_record = nusc.get('sample_data', cam_token)
        egocam = nusc.get('ego_pose', cam_record['ego_pose_token'])
        cam = nusc.get('calibrated_sensor',
                       cam_record['calibrated_sensor_token'])

        cam_from_egocam = parse_pose(cam, inv=True)
        egocam_from_world = parse_pose(egocam, inv=True)

        E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
        I = cam['camera_intrinsic']

        full_path = Path(nusc.get_sample_data_path(cam_token))
        image_path = str(full_path.relative_to(nusc.dataroot))

        # cam_channels.append(cam_channel)
        intrinsics.append(I)
        extrinsics.append(E.tolist())
        images.append(image_path)

    return {
        'scene': scene_name,
        'token': sample_record['token'],

        'pose': world_from_egolidarflat.tolist(),
        'pose_inverse': egolidarflat_from_world.tolist(),

        'cam_ids': list(camera_rig),
        'cam_channels': cam_channels,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'images': images,
    }


def parse_scene(scene_record, nusc, camera_rigs=[[0, 1, 2, 3, 4, 5]]):
    data = []
    sample_token = scene_record['first_sample_token']

    while sample_token:
        sample_record = nusc.get('sample', sample_token)

        for camera_rig in camera_rigs:
            data.append(parse_sample_record(sample_record,
                        camera_rig, scene_record['name'], nusc))

        sample_token = sample_record['next']

    return data


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    default_config_path = Path(
        __file__).parents[0] / 'configs/config_generate_metadata.yaml'
    try:
        config = load_config(sys.argv[1])
    except IndexError:
        config = load_config(default_config_path)

    dataset_dir = config['dataset_dir']
    version = config['version']
    out_dir = config['out_dir']

    nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=True)

    for scene in tqdm(nusc.scene):
        data = parse_scene(scene, nusc)
        os.makedirs(out_dir, exist_ok=True)
        with open(Path(out_dir) / f'{scene["name"]}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    print('Metadata generation complete!')

if __name__ == '__main__':
    main()