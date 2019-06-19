from __future__ import division

from models import Darknet
from utils.loggerx import *
from utils.utils import *
from utils.parse_config import *
from utils.data_loader import *
from terminaltables import AsciiTable

import os
import time
import datetime
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

# debug cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolo_cars")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-cars.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/cars.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--restart_point", type=int, default=None, help="checkpoint to restart epoch training")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, type=bool, help="if True computes mAP every evaluation")
    parser.add_argument("--multiscale_training", default=True, type=bool, help="allow for multi-scale training")
    parser.add_argument("--conf_thres", default=0.1, type=float, help="confidence threshold for nms")
    parser.add_argument("--nms_thres", default=0.5, type=float, help="nsm threshold to reduce # of boxes")
    parser.add_argument("--iou_thres", default=0.5, type=float, help="iou threshold")
    parser.add_argument("--optimizer", default="Adam", type=str, help="pytorch optimizer")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="< 0.001 or nms will freeze")
    opt = parser.parse_args()
    print(opt)

    cwd = os.getcwd()
    # variables
    model_run_name = opt.model_name
    model_def = opt.model_def
    data_cfg = opt.data_config
    pretrained_weights = opt.pretrained_weights
    output_path = "%s/%s/" %(cwd, model_run_name)

    epochs = opt.epochs
    batch_size = opt.batch_size
    gradient_accumulations = opt.gradient_accumulations
    n_cpu = opt.n_cpu
    img_size = opt.img_size
    compute_map = opt.compute_map
    multiscale_training = opt.multiscale_training

    if opt.checkpoint_interval is None:
        checkpoint = False      # boolean to save model checkpoint
    else:
        checkpoint = True
        checkpoint_interval = opt.checkpoint_interval
    if opt.evaluation_interval is None:
        evaluate = False
    else:
        evaluate = True        # boolean to run evaluation
        evaluation_interval = opt.evaluation_interval
    if opt.restart_point is None:
        start_i = 1
    else:
        start_i = opt.restart_point + 1

    ## HYPERPARAMETERS
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    iou_thres = opt.iou_thres
    optimizer_name = opt.optimizer
    learning_rate = opt.learning_rate

    # initialize logger
    logger = Logger("%s/logs/" %(output_path))

    # check for gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print("device: ", device)

    # make output directories
    if not os.path.exists("%s/outputs" %(output_path)):
        os.makedirs("%s/outputs" %(output_path), exist_ok=True)
    if not os.path.exists("%s/checkpoints" %(output_path)):
        os.makedirs("%s/checkpoints" %(output_path), exist_ok=True)
    if not os.path.exists("%s/logs" %(output_path)):
        os.makedirs("%s/logs" %(output_path), exist_ok=True)

    # dump argparse variables to parameter.txt file
    parameters = json.dumps(vars(opt), indent=4)
    with open("%s/outputs/parameters.txt" %(output_path), "w") as fp:
        json.dump(parameters, fp)

    # Get data configuration
    data_config = parse_data_config(data_cfg)
    train_path = "%s/%s" %(cwd, data_config["train"])
    valid_path = "%s/%s" %(cwd, data_config["valid"])
    class_names = load_classes("%s/%s" %(cwd, data_config["names"]))

    # Initiate model
    model = Darknet(model_def)
    model = model.to(device)
    model.apply(weights_init_normal)

    ## feature extraction if True (update and reshape last layers)
    feature_extract = False
    if feature_extract:
        # freeze model weights - feature extraction if requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

    # If specified we start from checkpoint
    if pretrained_weights:
        if pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained_weights))
        else:
            model.load_darknet_weights(pretrained_weights)

    ### ADD IMAGE TRANSFORMS HERE
    transforms_list = [

    ]

    # Get train dataloader
    print("\nLoad training data... ")
    train_dataset = BoxImageDataset(train_path, transforms_list=[], use_pad=True, img_size=img_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=False,
        collate_fn=train_dataset.collate_fn
    )

    print("\nLoad validation path...")
    valid_dataset = BoxImageDataset(valid_path, transforms_list=[], use_pad=True, img_size=img_size)

    print("Using optimizer %s" %(optimizer_name))
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.ReLU(model.parameters(), lr=learning_rate)

    ## metrics
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(start_i, epochs+1):
        start_time = time.time()
        model.train()
        total_loss = []

        for batch_i, (filename, imgs, targets) in enumerate(train_dataloader):
            print("\ntrain batch: ", batch_i)
            batches_done = len(train_dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # print("imgs: ", imgs.shape)
            # print("targets: ", targets.shape)

            loss, outputs = model(imgs, targets)
            loss.backward()
            # add loss over all batches
            total_loss.append(loss.item())

            if batches_done % gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            print("batch training time: ", time.time() - start_time)

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(train_dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        print("train time: ", time.time() - start_time)

        # ----------------
        #   Evaluate
        # ----------------

        if evaluate:
            eval_start_time = time.time()

            if epoch % evaluation_interval == 0:
                print("\n\n---- Evaluating Model ----")
                print("\nLoad validation data... ")
                valid_dataloader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=n_cpu,
                    collate_fn=valid_dataset.collate_fn
                )

                with torch.no_grad():
                    # Evaluate the model on the validation set
                    model.eval()

                    valid_labels = []
                    sample_metrics = []  # List of tuples (TP, confs, pred)
                    img_paths = []  # Stores image paths
                    img_detections = []  # Stores detections for each image index

                    for batch_i, (valid_filename, valid_imgs, valid_targets) in enumerate(valid_dataloader):
                        print("\nevaluating batch: ", batch_i)
                        # print(valid_filename)

                        valid_imgs = Variable(valid_imgs.to(device), requires_grad=False)

                        # Extract labels
                        batch_labels = valid_targets[:, 1].tolist()
                        valid_labels += batch_labels
                        print("# valid_targets ", len(valid_targets))
                        if len(valid_targets) > 0:
                            for target in valid_targets:
                                print(class_names[int(target[1].item())])

                        # Rescale target -> 0, class, x1, y1, x2, y2
                        rescale_targets = xywh2xyxy(valid_targets[:, 2:]) * img_size
                        valid_targets[:, 2:] = rescale_targets
                        # print("rescaled targets: ", rescale_targets)

                        outputs = model(valid_imgs).detach()
                        print("# outputs: ", outputs[0].shape)
                        # nms output -> (x1, y1, x2, y2, object_conf, class_score, class_pred)
                        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
                        if outputs[0] is None:
                            print("nms outputs is None")
                            continue
                        print("# nms outputs: ", outputs[0].shape)

                        print("%s metrics: " %(valid_filename))
                        # batch_metrics -> true_positives, pred_scores, pred_labels
                        batch_metrics = get_batch_statistics(outputs, valid_targets, iou_threshold=iou_thres)
                        sample_metrics += batch_metrics

                        b_precision, b_recall, b_AP, b_f1, b_ap_class = ap_per_class(batch_metrics[0][0], batch_metrics[0][1],\
                                                                                   batch_metrics[0][2], batch_labels)
                        print("b_precision: %s, b_recall: %s, b_AP: %s, b_f1: %s, b_ap_class: %s" \
                              %(b_precision, b_recall, b_AP, b_f1, b_ap_class))

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
                        evaluation_metrics.append(("%s_mAP" %(class_names[c]), AP[i].mean()))
                    logger.list_of_scalars_summary(evaluation_metrics, epoch)

                    # Print class APs and mAP
                    ap_table = [["Index", "Class name", "AP", "mAP"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i], "%.5f" %AP[i].mean()]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- Validation mAP {AP.mean()}\n")
                    print("evaluation time: ", time.time() - eval_start_time)

                    with open("%s/outputs/validation_maps.csv" %(output_path), "a") as fp:
                        fp.write("%s, %s\n" %(epoch, AP.mean()))

        if checkpoint:
            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), "%s/checkpoints/yolov3_ckpt_%d.pth" %(output_path, epoch))

        print("epoch time: ", time.time() - start_time)
