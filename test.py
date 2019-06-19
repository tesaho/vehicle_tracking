from __future__ import division

from models import Darknet
from utils.utils import *
from utils.data_loader import *
from utils.parse_config import *
from terminaltables import AsciiTable

import os
import time
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

# debug cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolo_cars")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-cars.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/cars.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--conf_thres", default=0.01, type=float, help="confidence threshold for nms")
    parser.add_argument("--nms_thres", default=0.5, type=float, help="nsm threshold to reduce # of boxes")
    parser.add_argument("--iou_thres", default=0.5, type=float, help="iou threshold")
    opt = parser.parse_args()
    print(opt)

    cwd = os.getcwd()
    # variables
    model_run_name = opt.model_name
    model_def = opt.model_def
    data_cfg = opt.data_config
    pretrained_weights = opt.pretrained_weights
    output_path = "%s/%s/" %(cwd, model_run_name)

    batch_size = opt.batch_size
    n_cpu = opt.n_cpu
    img_size = opt.img_size

    ## HYPERPARAMETERS
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    iou_thres = opt.iou_thres

    # check for gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print("device: ", device)

    # make output directories
    if not os.path.exists("%s/eval" %(output_path)):
        os.makedirs("%s/eval" %(output_path), exist_ok=True)

    # dump argparse variables to parameter.txt file
    with open("%s/eval/parameters.txt" %(output_path), "w") as fp:
        json.dump(vars(opt), fp)

    # Get data configuration
    data_config = parse_data_config(data_cfg)
    valid_path = "%s/%s" %(cwd, data_config["valid"])
    class_names = load_classes("%s/%s" %(cwd, data_config["names"]))

    # Initiate model
    model = Darknet(model_def)
    model = model.to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if pretrained_weights:
        if pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained_weights))
            checkpoint_name = pretrained_weights.split("/")[-1].split(".")[0]
        else:
            model.load_darknet_weights(pretrained_weights)
            checkpoint_name = "darknet_weights"
        print("checkpoint_name: ", checkpoint_name)

    # Get data
    print("\nLoad validation path...")
    valid_dataset = BoxImageDataset(valid_path, transforms_list=[], use_pad=True, img_size=img_size)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_cpu,
        collate_fn=valid_dataset.collate_fn
    )

    # ----------------
    #   Evaluate
    # ----------------

    with torch.no_grad():
        # Evaluate the model on the validation set
        model.eval()

        valid_labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        img_paths = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        batch_precision = []
        batch_recall = []
        batch_ap = []
        batch_map = []
        batch_f1 = []
        batch_class = []
        batch_class_names = []
        batch_ap_class = []
        batch_scores = []
        batch_preds = []
        batch_conf = []

        for batch_i, (valid_filename, valid_imgs, valid_targets) in enumerate(valid_dataloader):
            print("\nevaluating batch: ", batch_i)

            valid_imgs = Variable(valid_imgs.to(device), requires_grad=False)

            # Extract labels
            batch_labels = valid_targets[:, 1].tolist()
            valid_labels += batch_labels
            print("# valid_targets ", len(valid_targets))
            if len(valid_targets) > 0:
                for target in valid_targets:
                    print(class_names[int(target[1].item())])
            b_labels = list(np.unique(batch_labels))
            b_labels.sort()
            batch_class_names += [class_names[int(x)] for x in b_labels]
            batch_class += [int(x) for x in b_labels]
            img_paths += [valid_filename[0]] * len(b_labels)

            # Rescale target -> 0, class, x1, y1, x2, y2
            rescale_targets = xywh2xyxy(valid_targets[:, 2:]) * img_size
            valid_targets[:, 2:] = rescale_targets

            # Predictions
            outputs = model(valid_imgs).detach()
            print("# outputs: ", outputs[0].shape)
            # nms output -> (x1, y1, x2, y2, object_conf, class_score, class_pred)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            if outputs[0] is None:
                print("nms outputs is None")
                continue
            print("# nms outputs: ", outputs[0].shape)
            img_detections.extend(outputs)
            img = np.array(Image.open(valid_filename[0]))
            output_list = []
            for output in outputs:
                print(output.shape)
                batch_preds += output[:, -1].tolist()
                batch_scores += output[:, -2].tolist()
                batch_conf += output[:, -3].tolist()
                # unpad x1, y1, x2, y2
                rescale_outputs = rescale_boxes(output, opt.img_size, img.shape[:2])
                # print(rescale_outputs)
                x1 = rescale_outputs[:, 0].tolist()
                y1 = rescale_outputs[:, 1].tolist()
                x2 = rescale_outputs[:, 2].tolist()
                y2 = rescale_outputs[:, 3].tolist()
                box_w = np.array(x2) - np.array(x1)
                box_h = np.array(y2) - np.array(y1)
                output_list.extend([[valid_filename[0]]*len(x1), x1, y1, x2, y2, box_w, box_h])
            detections = pd.DataFrame(data=output_list).T
            detections.columns = ["filename", "x1", "y1", "x2", "y2", "w", "h"]
            detections["obj_conf"] = output[:, -3].tolist()
            detections["score"] = output[:, -2].tolist()
            detections["pred"] = output[:, -1].tolist()
            detections.to_csv("%s/eval/%s_predictions_iou_%s.csv" % (output_path, checkpoint_name, iou_thres))

            # batch_metrics -> true_positives, pred_scores, pred_labels
            batch_metrics = get_batch_statistics(outputs, valid_targets, iou_threshold=iou_thres)
            sample_metrics += batch_metrics
            # ap class per image
            b_precision, b_recall, b_AP, b_f1, b_ap_class = ap_per_class(batch_metrics[0][0], batch_metrics[0][1],\
                                                                       batch_metrics[0][2], batch_labels)
            print("b_precision: %s, b_recall: %s, b_AP: %s, b_f1: %s, b_ap_class: %s" \
                  %(b_precision, b_recall, b_AP, b_f1, b_ap_class))
            batch_precision += b_precision.tolist()
            batch_recall += b_recall.tolist()
            batch_ap += b_AP.tolist()
            batch_f1 += b_f1.tolist()
            batch_ap_class += b_ap_class.tolist()

        # create csv of batch results
        batch_results = pd.DataFrame(data=[img_paths, batch_class, batch_class_names,
                                             batch_precision, batch_recall,
                                             batch_ap, batch_f1, batch_conf,
                                             batch_scores, batch_preds]).T
        batch_results.columns = ["filename", "class", "class_names", "precision",
                                              "recall", "ap",  "f1", "obj_conf", "score", "pred"]
        batch_results.to_csv("%s/eval/%s_batch_results_iou_%s.csv" \
                             %(output_path, checkpoint_name, iou_thres))

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, valid_labels)

        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        for i, c in enumerate(ap_class):
            evaluation_metrics.append(("%s_mAP" % (class_names[c]), AP[i].mean()))

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP", "mAP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i], "%.5f" %AP[i].mean()]]
        print(AsciiTable(ap_table).table)
        print(f"---- Evaluation mAP {AP.mean()}\n")

        # output check point precision, recall, map, and f1
        with open("%s/eval/%s_map_iou_%s.csv" %(output_path, checkpoint_name, iou_thres), "w") as fp:
            for i in range(len(evaluation_metrics)):
                fp.write("%s, %s\n" %(evaluation_metrics[i][0], evaluation_metrics[i][1]))





