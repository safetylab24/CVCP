from ..utils import box_torch_ops
import torch

num_point = 5
NMS_POST_MAXSIZE = 500


class TwoStageDetectorUtils():
    @staticmethod
    def combine_loss(one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    @staticmethod
    def get_box_center(boxes):
        # box [List]
        centers = []
        for box in boxes:
            if num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])

            elif num_point == 5:
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                corners = box_torch_ops.center_to_corner_box2d(
                    center2d, dim2d, rotation_y)

                front_middle = torch.cat(
                    [(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat(
                    [(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat(
                    [(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat(
                    [(corners[:, 1] + corners[:, 2])/2, height], dim=-1)

                points = torch.cat([box['box3d_lidar'][:, :3], front_middle, back_middle, left_middle,
                                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    @staticmethod
    def reorder_first_stage_pred_and_feature(first_pred, label, features):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1]
        feature_vector_length = sum([feat[0].shape[-1] for feat in features])

        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size,
                                                       NMS_POST_MAXSIZE, box_length
                                                       ))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size,
                                                        NMS_POST_MAXSIZE
                                                        ))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size,
                                                             NMS_POST_MAXSIZE), dtype=torch.long
                                                            )
        roi_features = features[0][0].new_zeros((batch_size,
                                                 NMS_POST_MAXSIZE, feature_vector_length
                                                 ))

        for i in range(batch_size):
            num_obj = features[0][i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['box3d_lidar']

            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
            box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            roi_features[i, :num_obj] = torch.cat(
                [feat[i] for feat in features], dim=-1)

        label['rois'] = rois
        label['roi_labels'] = roi_labels
        label['roi_scores'] = roi_scores
        label['roi_features'] = roi_features

        label['has_class_labels'] = True

        return label

    @staticmethod
    def post_process(batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = []

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            # this is the predicted iou
            cls_preds = batch_dict['batch_cls_preds'][index]
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1)
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(
                cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # currently don't need nms
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
            }

            pred_dicts.append(pred_dict)

        return pred_dicts
