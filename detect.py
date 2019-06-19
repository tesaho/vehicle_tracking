from __future__ import division

from models import *
from utils.utils import *
from utils.data_loader import *

import os
import time
import datetime
import argparse
import random

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolo_cars")
    parser.add_argument("--image_folder", type=str, default="data/cars/valid_images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-cars.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/cars/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="threshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cwd = os.getcwd()
    model_name = opt.model_name
    iou_thres = opt.iou_thres
    output_path = "%s/%s/" % (cwd, opt.model_name)

    # create img box directory
    img_box_path = "%s/img_boxes_iou_%s/" %(output_path, iou_thres)
    if not os.path.exists(img_box_path):
        os.makedirs(img_box_path, exist_ok=True)
    # create map_eval directory
    detections_path = "%s/detections_iou_%s/" %(output_path, iou_thres)
    if not os.path.exists(detections_path):
        os.makedirs(detections_path, exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        print("loading pretrained weights: ", opt.pretrained_weights)
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    with torch.no_grad():
        model.eval()  # Set in evaluation mode
        dataset = ImageDataset(opt.image_folder, use_pad=True, img_size=opt.img_size)
        dataloader = DataLoader(
            # ImageFolder(opt.image_folder, img_size=opt.img_size),
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu
        )

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores images
        paths = [] # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        inference_start_time = time.time()
        for batch_i, (img_paths, imgs_aug) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(imgs_aug.type(Tensor))

            # Get detections
            detections = model(input_imgs)
            print("# outputs: ", detections[0].shape)
            outputs = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            if outputs[0] is None:
                print("nms outputs is None")
                continue
            print("# nms outputs: ", outputs[0].shape)

            # Log progress
            inference_time = datetime.timedelta(seconds=time.time() - inference_start_time)
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(imgs_aug)
            paths.extend(img_paths)
            img_detections.extend(outputs)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, outputs) in enumerate(zip(paths, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))
            filename = path.split("/")[-1].replace('jpg', 'txt')

            # use augmented image from tensor (channel, height, weight) to np.array(PIL) (height, width, channel)
            img_aug = np.array(imgs[img_i].permute(1, 2, 0))

            print("img_aug: ", img_aug.shape)

            # get original image
            img_og = np.array(Image.open(path))
            print("img_og: ", img_og.shape)

            # Create plot
            fig = plt.figure()
            plt.imshow(img_og)

            # create img_box file
            filename = path.split("/")[-1].replace('jpg', 'txt')
            fp = open("%s/%s" %(detections_path, filename), "w+")

            # Draw bounding boxes and labels of detections
            if outputs is not None:
                # Rescale boxes to original image
                rescale_outputs = rescale_boxes(outputs, opt.img_size, img_og.shape[:2])
                unique_labels = rescale_outputs[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                # print("n_cls_preds: ", n_cls_preds)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_score, cls_pred in rescale_outputs:

                    if cls_score.item() < iou_thres:
                        continue

                    print("\t+ Label: %s, Score: %.5f" % (classes[int(cls_pred)], cls_score.item()))
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax = plt.gca()
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

                    # write boxes to file
                    box_list = [classes[int(cls_pred.item())],
                                "%.5f" %cls_score.item(),
                                int(x1.item()), int(y1.item()),
                                int(box_w.item()), int(box_h.item())]
                    # print("box_list: ", box_list)
                    for x in box_list:
                        fp.write(str(x))
                        fp.write(" ")
                    fp.write("\n")

            # close detections writer
            fp.close()

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig("%s/%s.png" %(img_box_path, filename), bbox_inches="tight", pad_inches=0.0)
            plt.close()
