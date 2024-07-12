import numpy as np

def load_annotations(result: dict, info: dict):
    if "gt_boxes" in info:
        gt_boxes = np.array(info["gt_boxes"]).astype(np.float32)
        gt_boxes[np.isnan(gt_boxes)] = 0
        result["annotations"] = {
            "boxes": gt_boxes,
            "names": info["gt_names"],
            "tokens": info["gt_boxes_token"],
            "velocities": np.array(info["gt_boxes_velocity"]).astype(np.float32),
        }
    else:
        raise ValueError("No gt_boxes in info")

    return result