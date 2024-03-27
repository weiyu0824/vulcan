from typing import List
import torch

def calculate_iou(box1, box2):
    """Calculate intersection over union (IoU) between two bounding boxes."""
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    
    # Calculate coordinates of intersection
    x_intersection = max(x1a, x1b)
    y_intersection = max(y1a, y1b)
    w_intersection = max(0, min(x2a, x2b) - x_intersection)
    h_intersection = max(0, min(y2a, y2b) - y_intersection)
    
    # Calculate area of intersection
    intersection_area = w_intersection * h_intersection
    
    # Calculate area of union
    area_box1 = (x2a - x1a) * (y2a - y1a)
    area_box2 = (x2b - x1b) * (y2b - y1b)
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou


def calculate_map(true_boxes, pred_boxes, iou_threshold=0.5, num_classes=1, eps=10e-5):
    """Calculate mean average precision (mAP) for object detection.
        
    """
    # box_format = [x1, y1, x2, y2, cls, conf]
    average_precisions = []

    # num_classes = 1
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for pred_box in pred_boxes:
            if pred_box[4] == c:
                detections.append(pred_box)
        
        for true_box in true_boxes:
            if true_box[4] == c:
                ground_truths.append(true_box)

        detections.sort(key=lambda x: x[5], reverse=True) #sort detection by confidence (descending)
        TP = torch.zeros(len(pred_boxes)) 
        FP = torch.zeros(len(pred_boxes))
        total_true_bboxes = len(ground_truths)
        detected = [False for _ in range(len(ground_truths))] 
        # breakpoint()
        for detection_idx, detection in enumerate(detections):
           # Find what gt box does this detection match
            best_idx, best_iou = -1, 0
            for ground_truth_idx, ground_truth in enumerate(ground_truths):
                iou = calculate_iou(ground_truth[:4], detection[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = ground_truth_idx
            
            # if best_idx == -1:
            # breakpoint()
            # print('bug:', len(ground_truths), c,  len(detections), best_idx)
            # Check if this detection is FP or TP
            if best_iou > iou_threshold and detected[best_idx] == False:
               TP[detection_idx] = 1
            else:
               FP[detection_idx] = 1

        # [1, 1, 0, 0, 1] = [1, 2, 2, 2, 3]
        # Technique to speed up
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps) 
        recalls = TP_cumsum / (total_true_bboxes + eps)

        # ?num_classes=1, )?
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precisions.append(torch.trapz(precisions, recalls)) 
        # print('cls', c, precisions, recalls, TP_cumsum, FP_cumsum)
    # print(average_precisions)
    return  sum(average_precisions) / len(average_precisions)