from det3d.models.bbox_heads.center_head import CenterHead
import logging
import torch

# Define your tasks
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

# Initialize the logger
logger = logging.getLogger("CenterHead")
logging.basicConfig(level=logging.INFO)

# Configuration for the CenterHead
bbox_head_config = dict(
    type="CenterHead",
    in_channels=sum([256, 256]),
    tasks=tasks,
    dataset='nuscenes',
    weight=0.25,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
    common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2)},
    share_conv_channel=64,
    dcn_head=False
)

# Initialize the CenterHead model
center_head = CenterHead(
    in_channels=bbox_head_config['in_channels'],
    tasks=bbox_head_config['tasks'],
    dataset=bbox_head_config['dataset'],
    weight=bbox_head_config['weight'],
    code_weights=bbox_head_config['code_weights'],
    common_heads=bbox_head_config['common_heads'],
    logger=logger,
    share_conv_channel=bbox_head_config['share_conv_channel'],
    dcn_head=bbox_head_config['dcn_head']
)

# Print the model architecture
print(center_head)
