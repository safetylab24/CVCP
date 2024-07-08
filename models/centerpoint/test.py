from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version='v1.0-trainval', dataroot='/home/vrb230004/media/datasets/nuscenes2', verbose=False)

target_sample_sensor_data_filename = 'n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530649557'

sensor_name = 'LIDAR_TOP'
for sample in nusc.sample:
    sample_data = sample['data']
    sample_sensor_data = nusc.get('sample_data', sample_data[sensor_name])
    
    if target_sample_sensor_data_filename in sample_sensor_data['filename']:
        scene = nusc.get('scene', sample['scene_token'])
        print(scene['name'])
        break
    else:
        print("not founc BROOOOOHrgeu")