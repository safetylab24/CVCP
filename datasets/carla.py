import json
import math
import os
import random

import torchvision
from torch.utils.data import Subset, DataLoader

from tools.geometry import *
from PIL import Image, ImageFile


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, set, pos_class):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.set = set  # 'train', 'val', 'test'
        self.pos_class = pos_class

        self.data_path = data_path

        self.label_root = os.path.join(
            self.data_path, 'labels_' + self.set, 'agents/')

        self.ticks = len(os.listdir(os.path.join(
            self.data_path, self.set, 'agents/0/back_camera')))
        self.agent_num = len(os.listdir(
            os.path.join(self.data_path, self.set, 'agents/')))
        self.offset = 0

        self.to_tensor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution, bev_start_position, bev_dimension
        )

    def get_input_data(self, index, agent_path):
        images = []
        intrinsics = []
        extrinsics = []

        with open(os.path.join(agent_path, 'sensors.json'), 'r') as f:
            sensors = json.load(f)

        for sensor_name, sensor_info in sensors['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":

                image = Image.open(os.path.join(
                    agent_path + sensor_name, f'{index}.png'))

                intrinsic = torch.tensor(sensor_info["intrinsic"])

                intrinsic = np.identity(3)
                sensor_opts = sensor_info['sensor_options']
                intrinsic[0, 2] = sensor_opts['image_size_x'] / 2.0
                intrinsic[1, 2] = sensor_opts['image_size_y'] / 2.0
                intrinsic[0, 0] = sensor_opts['image_size_x'] / \
                    (2.0 * np.tan(sensor_opts['fov'] * np.pi / 360.0))
                intrinsic[1, 1] = intrinsic[0, 0]

                intrinsic = torch.FloatTensor(intrinsic)

                translation = np.array(sensor_info["transform"]["location"])
                rotation = sensor_info["transform"]["rotation"]

                rotation[0] += 90
                rotation[2] -= 90

                r = Rotation.from_euler('zyx', rotation, degrees=True)

                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = r.as_matrix()
                extrinsic[:3, 3] = translation
                extrinsic = np.linalg.inv(extrinsic)

                normalized_image = self.to_tensor(image)

                images.append(normalized_image)
                intrinsics.append(intrinsic)
                extrinsics.append(torch.tensor(extrinsic))
                image.close()

        images, intrinsics, extrinsics = (torch.stack(images, dim=0),
                                          torch.stack(intrinsics, dim=0),
                                          torch.stack(extrinsics, dim=0))

        return images, intrinsics, extrinsics

    def get_label(self, agent_num, index=None):
        # label_r = Image.open(os.path.join(self.label_root, str(
        #     agent_num), self.pos_class, f'{index}.png'))
        # label = np.array(label_r)
        # label_r.close()
        
        label_r = cv2.imread(os.path.join(self.label_root, str(
            agent_num), self.pos_class, f'{index}.png'))
        label_r = cv2.cvtColor(label_r, cv2.COLOR_BGR2GRAY)

        label = np.array(label_r)
        label[label == 255] = 1

        background = np.ones(self.bev_dimension[:2])

        # road = mask(label, (128, 64, 128))
        # lane = mask(label, (157, 234, 50))
        # vehicles = mask(label, (0, 0, 142))

        # if np.sum(vehicles) < 5:
        #     lane = mask(label, (50, 234, 157))
        #     vehicles = mask(label, (142, 0, 0))

        if self.pos_class == 'vehicle':
            # add secondary center label
            # center_img = cv2.imread(os.path.join(self.label_root, str(
            # agent_num), 'center', f'{index}.png'))
            # center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
            # center_img[center_img==255] = 1
            # label = np.stack((label, center_img))
            
            background[label == 1] = 0

            label = np.stack((label, background))            
            label = label.astype(np.float32)

        elif self.pos_class == 'road':
            background[label == 1] = 0
            label = np.stack((label, background))

        #     # this is a refinement step to remove some impurities in the label caused by small objects
        #     road = (road * 255).astype(np.uint8)
        #     kernel_size = 2

        #     kernel = np.ones((kernel_size, kernel_size), np.uint8)

        #     road = cv2.dilate(road, kernel, iterations=1)
        #     road = cv2.erode(road, kernel, iterations=1)

        #     label = np.stack((road, empty))
        # elif self.pos_class == 'lane':
        #     empty[lane == 1] = 0

        #     label = np.stack((lane, empty))
        # elif self.pos_class == 'all':
        #     empty[vehicles == 1] = 0
        #     empty[lane == 1] = 0
        #     empty[road == 1] = 0
        #     label = np.stack((vehicles, road, lane, empty))
        # label = np.stack((label, empty))
        return label

    def __len__(self):
        return 4625 if self.set == 'val' else self.ticks*self.agent_num 

    def __getitem__(self, index):
        agent_number = math.floor(index / self.ticks)
        agent_path = os.path.join(
            self.data_path, self.set, f"agents/{agent_number}/")
        index = (index + self.offset) % self.ticks

        images, intrinsics, extrinsics = self.get_input_data(index, agent_path)
        labels = self.get_label(agent_number, index)

        return images, intrinsics, extrinsics, labels


def compile_data(set, dataroot, pos_class, batch_size=4, num_workers=4, seed=0):
    dataset = CarlaDataset(dataroot, set, pos_class)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False)

    return loader
