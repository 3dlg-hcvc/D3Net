from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
import re

EVALUATION_TYPES = {

    "zt_wo_d": 3,
    "zt_w_d": 4,
    "st_wo_d": 0,
    "st_w_d": 1,
    "mt": 2,

}

# SEM_LABELS = {3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf',
#               11: 'picture',
#               12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator', 28: 'shower curtain', 33: 'toilet',
#               34: 'sink',
#               36: 'bathtub', 39: 'otherfurniture'}


def get_aabb_iou(box_1, box_2):
    """ Compute 3D axis-aligned bounding box IoU.
    Input:
        box_1: numpy array (8,3), assume up direction is Z
        box_2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D axis-aligned bounding box IoU
    """
    box_1_x_min, box_1_y_min, box_1_z_min = box_1.min(axis=0)
    box_1_x_max, box_1_y_max, box_1_z_max = box_1.max(axis=0)
    box_2_x_min, box_2_y_min, box_2_z_min = box_2.min(axis=0)
    box_2_x_max, box_2_y_max, box_2_z_max = box_2.max(axis=0)

    x_a = np.maximum(box_1_x_min, box_2_x_min)
    y_a = np.maximum(box_1_y_min, box_2_y_min)
    z_a = np.maximum(box_1_z_min, box_2_z_min)
    x_b = np.minimum(box_1_x_max, box_2_x_max)
    y_b = np.minimum(box_1_y_max, box_2_y_max)
    z_b = np.minimum(box_1_z_max, box_2_z_max)

    intersection_volume = np.maximum((x_b - x_a), 0) * np.maximum((y_b - y_a), 0) * np.maximum((z_b - z_a), 0)

    if intersection_volume == 0:
        # in case both box volumes are zero
        return 0

    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume)
    return iou


def match_gt_pred(cost_matrix):
    # apply hungarian algorithm
    return linear_sum_assignment(cost_matrix)


def evaluate_one_query(pred_info, gt_info):
    pred_bboxes_count = len(pred_info["aabbs"])
    gt_bboxes_count = len(gt_info["aabbs"])

    # initialize true positives
    iou_25_tp = 0
    iou_50_tp = 0

    # initialize the cost matrix
    square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
    iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len))
    for pred_aabb_idx, pred_aabb in enumerate(pred_info["aabbs"]):
        for gt_aabb_idx, gt_aabb in enumerate(gt_info["aabbs"]):
            iou_matrix[pred_aabb_idx, gt_aabb_idx] = get_aabb_iou(np.array(gt_aabb), np.array(pred_aabb))

    # apply matching algorithm
    row_idx, col_idx = match_gt_pred(iou_matrix * -1)

    # iterate matched pairs, check ious
    for i in range(pred_bboxes_count):
        iou = iou_matrix[row_idx[i], col_idx[i]]
        # calculate true positives
        if iou >= 0.25:
            iou_25_tp += 1
        if iou >= 0.5:
            iou_50_tp += 1

    # calculate precision, recall and f1-score for the current scene
    iou_25_f1_score = 2 * iou_25_tp / (pred_bboxes_count + gt_bboxes_count)
    iou_50_f1_score = 2 * iou_50_tp / (pred_bboxes_count + gt_bboxes_count)
    return iou_25_f1_score, iou_50_f1_score


def evaluate_one_zero_gt_query(pred_info):
    return 1 if len(pred_info["aabbs"]) == 0 else 0


def evaluate_all_scenes(all_pred_info, all_gt_info, scenes_info=None):
    all_gt_info_len = len(all_gt_info)

    #assert len(all_pred_info) == all_gt_info_len


    eval_type_mask = np.zeros(all_gt_info_len, dtype=np.uint8)
    # sem_label_class_mask = np.zeros(all_gt_info_len, dtype=np.uint16)
    iou_25_f1_scores = np.zeros(all_gt_info_len)
    iou_50_f1_scores = np.zeros(all_gt_info_len)

    for i, (key, value) in enumerate(tqdm(all_pred_info.items())):
        eval_type_mask[i] = all_gt_info[key]["eval_type"]
        # sem_label_class_mask[i] = all_gt_info[key]["sem_label"]
        if all_gt_info[key]["eval_type"] in (EVALUATION_TYPES["zt_wo_d"], EVALUATION_TYPES["zt_w_d"]):
            iou_25_f1_scores[i] = iou_50_f1_scores[i] = evaluate_one_zero_gt_query(value)
        else:
            iou_25_f1_scores[i], iou_50_f1_scores[i] = evaluate_one_query(value, all_gt_info[key])

    iou_25_results = {}
    iou_50_results = {}

    for sub_group in (3, 4, 0, 1, 2):
        selected_indices = eval_type_mask == sub_group
        if np.any(selected_indices):
            # micro-averaged scores of each semantic class and each subtype across queries
            iou_25_results[sub_group] = np.mean(iou_25_f1_scores[selected_indices])
            iou_50_results[sub_group] = np.mean(iou_50_f1_scores[selected_indices])
        else:

            iou_25_results[sub_group] = np.nan
            iou_50_results[sub_group] = np.nan

    # selected_indices = sem_label_class_mask == sem_class
    # if np.any(selected_indices):
    #     iou_25_results[sem_class]["overall"] = np.nanmean(iou_25_f1_scores[selected_indices])
    #     iou_50_results[sem_class]["overall"] = np.nanmean(iou_50_f1_scores[selected_indices])
    # else:
    #     iou_25_results[sem_class]["overall"] = np.nan
    #     iou_50_results[sem_class]["overall"] = np.nan
    #
    # iou_25_overall_overall_count.append(iou_25_results[sem_class]["overall"])
    # iou_50_overall_overall_count.append(iou_50_results[sem_class]["overall"])


    iou_25_results["overall"] = np.mean(iou_25_f1_scores)
    iou_50_results["overall"] = np.mean(iou_50_f1_scores)
    # for sub_group in EVALUATION_TYPES.values():
    #     selected_indices = eval_type_mask == sub_group
    #     if np.any(selected_indices):
    #         iou_25_results["overall"][sub_group] = np.nanmean(iou_25_f1_scores[selected_indices])
    #         iou_50_results["overall"][sub_group] = np.nanmean(iou_50_f1_scores[selected_indices])
    #     else:
    #         iou_25_results["overall"][sub_group] = np.nan
    #         iou_50_results["overall"][sub_group] = np.nan

    # iou_25_results["overall"] = np.nanmean(iou_25_overall_overall_count)
    # iou_50_results["overall"] = np.nanmean(iou_50_overall_overall_count)

    return iou_25_results, iou_50_results


def print_evaluation_results(title, iou_25_results, iou_50_results):
    print(f"{'=' * 100}\n|{'{:^98s}'.format(title)}|\n{'=' * 100}")
    print('{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}{5:<15}{6:<15}'.format("IoU", "zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt", "overall"))

    # hard code for statistics
    #groups = {0: [], 1: [], 2: [], 3: [], "overall": []}

    line_1_str = '{:<15}'.format("0.25")

    for sub_group_type, score in iou_25_results.items():
        line_1_str += '{:<15.1f}'.format(score * 100)
        # hard code for statistics
        #groups[sub_group_type].append(str(round(score, 3)))
    print(line_1_str)

    line_2_str = '{:<15}'.format("0.50")

    for sub_group_type, score in iou_50_results.items():
        line_2_str += '{:<15.1f}'.format(score * 100)
        # hard code for statistics
        # groups[sub_group_type].append(str(round(score, 3)))
    print(line_2_str)
    print(f"{'=' * 100}\n")

    latex1 = re.sub(' +', ' & ', line_1_str[15:])

    latex2 = re.sub(' +', ' & ', line_2_str[15:])


    print(latex1 + latex2)

