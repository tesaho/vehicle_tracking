from __future__ import division

import tqdm
import torch
from torch.nn import init
import numpy as np
import utils.map_eval as me

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    (remove padding)
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # print("og_h: %s og_w: %s img_h: %s img_w: %s" %(orig_h, orig_w, unpad_h, unpad_w))
    # Rescale bounding boxes to dimension of original image
    new_boxes = boxes.clone()
    new_boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    new_boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    new_boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    new_boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return new_boxes

def normalize_box(width, height, xmin, xmax, ymin, ymax):

    x = (xmin + xmax)/2. * 1./width
    w = (xmax - xmin) * 1./width
    y = (ymin + ymax)/2. * 1./height
    h = (ymax - ymin) * 1./height

    return (x, y, w, h)

def denormalize_box(width, height, x, y, w, h):
    """
    width, height = image width, height
    x = x_centered on scale (0,1)
    y = y_centered on scale (0,1)
    w, h = box width, height
    """

    xmax = int(width * (x + w / 2.0))
    xmin = int(width * (x - w / 2.0))
    ymax = int(height * (y + h / 2.0))
    ymin = int(height * (y - h / 2.0))

    return (xmin, ymin, xmax, ymax)

def xywh2xyxy(x):
    """
    x = (x_center, y_center, width, height)
    y = (xmin, ymin, xmax, ymax)
    """

    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y

def ap_per_image(outputs, targets, iou_threshold=0.5, method="every_point"):
    """
    outputs: x1, y1, x2, y2, obj conf, class conf, class
    targets: 0, label, x1, y1, x2, y2
    method: every_point, eleven_point

    :return:
    """

    # list containing metrics (precision, recall, average precision) of each class
    ret = []
    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    # groundTruths = []
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = outputs[0]

    # collect classes
    classes = [target[1] for target in targets]
    classes += [detection[-1] for detection in detections]
    classes = list(set(classes))
    classes.sort()
    print("classes: ", classes)

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        # Get only detection of class c
        dects = detections[detections[:, -1] == c]
        # Get only ground truths of class c
        gts = targets[targets[:, 1] == c]
        # number positives
        npos = len(gts)
        # sort detections by decreasing confidence obj confidence:4, class confidence:5
        dects = sorted(dects, key=lambda conf: conf[5], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with count of gts for each class {fname: # gt}
        # det = Counter([cc[1] for cc in gts])
        # for key, val in det.items():
        #     det[key] = np.zeros(val)
        det = np.zeros(len(gts))
        print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][:3],))
            iouMax = 0.0
            jmax = 0
            for j in range(len(gts)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                # iou(dect x1,y1,x2,y2  gt x1,y1,x2,y2)
                iou = bbox_iou(dects[d][:4].unsqueeze(0), gts[j][2:].unsqueeze(0))
                print(d, iou)
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
                    # print("iouMax: %s jmax: %s" %(iouMax, jmax))
            # Assign detection as true positive/don't care/false positive
            if iouMax >= iou_threshold:
                # value of det[fname][jmax]
                if det[jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("iouMax < iou_threshold: FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / (npos + 1.0e-16) # numerical stability
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        # Depending on the method, call the right implementation
        if method == "every_point":
            [ap, mpre, mrec, _] = me.calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = me.elevenPointInterpolatedAP(rec, prec)
        # add class result in the dictionary to be returned
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        # print(r)
        ret.append(r)

    return ret


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness descending index
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        # i = pred_cls == c
        # i index where pred_cls == c
        i = np.where(pred_cls == c)[0]
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        # print(output)
        pred_boxes = output[:, :4]
        pred_conf = output[:, 4]      # object_conf = 4, class_conf = 5, pred = 6
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        # what does this do?
        # targets -> 0, label, x, y, w, h
        # annotations -> label, x, y, w, h
        # annotations = targets[targets[:, 0] == sample_i][:, 1:]
        annotations = targets[:, 1:]
        # target_labels = annotations[:, 0] if len(annotations) else []
        target_labels = []
        if len(annotations) > 0:
            target_labels = annotations[:, 0]
        # print("target_labels: ", target_labels)

        if len(annotations) > 0:
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    print("targets found")
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    # print("pred_label not in target_labels")
                    continue

                # print("pred_box: ", pred_box.unsqueeze(0))
                # print("target_boxes: ", target_boxes)
                # returns the max iou value and the index position of the max iou value over dimension 0
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # print("iou: %s box_index: %s" %(iou.item(), box_index))
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_conf, pred_labels])
        # print("detected_boxes: ", detected_boxes)

    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            print("processing next image")
            continue
        # class confidence score = box confidence score * conditional class probability
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # print(detections.shape)
        # Perform non-maximum suppression
        keep_boxes = []
        i = 0
        print("performing nms...")
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            i += 1
            if i >= 11000:
                break
        # print("keep_boxes: ", keep_boxes)
        # if keep_boxes:
        output[image_i] = torch.stack(keep_boxes)
        # print("output: ", output)

        # memory release
        del detections
        del keep_boxes

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
