from nuscenes import NuScenes
import pickle
import numpy as np
import json
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
from pyquaternion import Quaternion

try:
    from nuscenes.utils import splits
except:
    print("nuScenes devkit not Found!")

general_to_detection = {
    "animal": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.wheelchair": "ignore",
    "movable_object.debris": "ignore",
    "movable_object.pushable_pullable": "ignore",
    "static_object.bicycle_rack": "ignore",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(
    nusc, sample_data_token: str, selected_anntokens: List[str] = None
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

CAM_CHANS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

def _fill_trainval_infos(nusc, train_scenes, test=False, filter_zero=True):
    scene_infos = {}

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.

    for sample in tqdm(nusc.sample):
        scene_token = sample["scene_token"]

        # Initialize scene infos if not already done
        if scene_token not in scene_infos:
            scene_infos[scene_token] = []

        # Get reference pose and timestamp
        ref_sd_token = sample["data"][ref_chan]
        _, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        info = {}
        
        
        annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)

        locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
        velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
        rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
        names = np.array([b.name for b in ref_boxes])
        tokens = np.array([b.token for b in ref_boxes])
        gt_boxes = np.concatenate(
            [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
        )

        assert len(annotations) == len(gt_boxes) == len(velocity) == len(names) == len(tokens)

        if not filter_zero:
            info["gt_boxes"] = gt_boxes.tolist()  # Convert to list for JSON serialization
            info["gt_boxes_velocity"] = velocity.tolist()  # Convert to list for JSON serialization
            info["gt_names"] = [general_to_detection[name] for name in names]
            info["gt_boxes_token"] = tokens.tolist()  # Convert to list for JSON serialization
        else:
            info["gt_boxes"] = gt_boxes[mask, :].tolist()  # Convert to list for JSON serialization
            info["gt_boxes_velocity"] = velocity[mask, :].tolist()  # Convert to list for JSON serialization
            # info["gt_names"] = [general_to_detection[name] for name in names if general_to_detection[name] != "ignore"]
            info["gt_names"] = names[mask].tolist()
            for i in range(len(info["gt_names"])):
                info["gt_names"][i] = general_to_detection[info["gt_names"][i]]
            info["gt_boxes_token"] = tokens[mask].tolist()  # Convert to list for JSON serialization

        assert len(info["gt_boxes"]) == len(info["gt_boxes_velocity"]) == len(info["gt_names"]) == len(info["gt_boxes_token"])
        if (len(info["gt_boxes"]) == 0) and filter_zero:
            continue
        scene_infos[scene_token].append((sample["token"], info))

    return scene_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def create_nuscenes_infos(root_path, version="v1.0-trainval", filter_zero=True):
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    scene_infos = _fill_trainval_infos(
        nusc, train_scenes, test, filter_zero=filter_zero
    )
    
    return scene_infos

if __name__ == "__main__":
    dataroot = '/home/vrb230004/media/datasets/nuscenes_test'
    version = 'v1.0-test'
    label_dir = '/home/vrb230004/CombinedModels/test_labels/'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    create_json = False
    create_pkl = True

    scene_infos = create_nuscenes_infos(
        dataroot, 
        version=version, 
        filter_zero=True)
    
    os.makedirs(label_dir, exist_ok=True)
    for scene_token, infos in tqdm(scene_infos.items()):
        scene_rec = nusc.get('scene', scene_token)
        scene_name = scene_rec['name']
        
        if create_json or create_pkl:
            scene_path_json = os.path.join(label_dir, scene_name + '.json')
            scene_path_pkl = os.path.join(label_dir, scene_name + '.pkl')
            # reformat infos so that scene_token is part of infos
            ref_infos = []
            for i in range(len(infos)):
                ref_info = infos[i][1]
                ref_info['token'] = infos[i][0]
                ref_infos.append(ref_info)
            
            if np.array(ref_infos[0]['gt_names']).shape[0] == np.array(ref_infos[0]['gt_boxes']).shape[0] and np.array(ref_infos[0]['gt_boxes']).shape[0] != 0:
                if create_pkl:
                    with open(scene_path_pkl, "wb") as f:
                        pickle.dump(ref_infos, f)
                if create_json:
                    with open(scene_path_json, "w") as f:
                        json.dump(ref_infos, f)
    
    print('Labels generated!')