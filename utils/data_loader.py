"""
data loader and augmentations
"""

import numpy as np
import PIL
import os
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import utils
from torch.utils.data import Dataset

# import images from a txt file with image paths
image_file_path = "./pytorch_models/PyTorch_YOLOv3/data/cars/train_small.txt"
img_size = 224


class BoxImageDataset(Dataset):
    """ Image dataset with bounding boxes """
    def __init__(self, image_file_path, transforms_list=[], use_pad=True, img_size=416):
        """
        image_file_path: path to text file with list of image file paths
        transform []: list of transforms to perform (padding done later)
        use_pad: Boolean default True, to use square padding
        img_size: resizing size for batches
        """
        self.image_file_path = image_file_path
        self.transforms = transforms_list      # list of all img transforms
        self.use_pad = use_pad      # boolean for using pad
        self.img_size = img_size    # resized img size

        with open(self.image_file_path, "r") as file:
            self.img_files = file.readlines()

    def __getitem__(self, index):
        """
        get an image
        """
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # convert to RGB file to PIL image to numpy array
        img = np.array(Image.open(img_path).convert("RGB"))
        height, width, channel = img.shape
        pad_size = max(width, height)

        # get bounding boxes
        bbs, labels = self.get_bounding_boxes(img_path, img)

        # augment bounding boxes and images
        if self.use_pad:
            box_transforms = [iaa.PadToFixedSize(width=pad_size, height=pad_size,
                              position="center", pad_cval=0)]
            box_transforms += self.transforms
        else:
            box_transforms = self.transforms
        # resize images to a specific size for batch stacking
        box_transforms += [iaa.Resize(self.img_size)]
        seq = iaa.Sequential(box_transforms)
        img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

        # convert to tensor
        img_aug = transforms.ToTensor()(img_aug)

        # convert labels back to x_center, y_center, w_norm, h_norm
        _, aug_h, aug_w = img_aug.shape
        targets = self.get_labels(bbs_aug, labels, aug_w, aug_h)

        return img_path, img_aug, targets

    def __len__(self):
        return len(self.img_files)

    def get_bounding_boxes(self, img_path, image):
        """
        bounding boxes
        """
        label_path = img_path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        # print("label_path: ", label_path)
        # PIL (width, height), np (height, width), torch (height, width)
        height, width, channel = image.shape

        bbs = None
        labels = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            labels = boxes[:, 0]
            # print("# boxes: ", len(boxes))

            bounding_boxes = []
            for i in range(len(boxes)):
                box = boxes[i]
                # boxes label_idx, x_center, y_center, width, height
                (xmin, ymin, xmax, ymax) = utils.denormalize_box(width, height, box[1], box[2], box[3], box[4])
                bounding_boxes.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax, label=box[0].data))
            # shape (np style (height, width))
            bbs = BoundingBoxesOnImage(bounding_boxes, shape=(height, width))

        return bbs, labels

    def get_labels(self, bbs, labels, pad_w, pad_h):
        """
        normalize bounding boxes into (0, labels, x_center, y_center, w_norm, h_norm)
        """
        # list of bounding boxes
        boxes = bbs.bounding_boxes
        targets = torch.zeros((len(boxes), 6))
        # targets -> 0, label, x_center, y_center, w, h
        targets[:, 1] = labels
        for i in range(len(boxes)):
            # normalize returns (x_center, y_center, w, h)
            targets[i, 2] = boxes[i].center_x / pad_w
            targets[i, 3] = boxes[i].center_y / pad_h
            targets[i, 4] = boxes[i].width / pad_w
            targets[i, 5] = boxes[i].height / pad_h

        return targets

    def collate_fn(self, batch):
        """
        custom collate function for DataLoader
        """

        # print("len(batch): ", len(batch))
        # create a list of batch functions
        paths = []
        mask_imgs = []
        targets = []
        for i in range(len(batch)):
            paths.append(batch[i][0])
            mask_imgs.append(batch[i][1].unsqueeze(0))
            boxes = batch[i][2]
            # Add sample index to targets
            boxes[:, 0] = i
            targets.append(boxes)
        targets = torch.cat(targets)
        mask_imgs = torch.cat(mask_imgs)
        # print("targets: ", targets)
        # print("targets size: ", targets.shape)
        # print("mask_imgs size: ", mask_imgs.shape)

        return paths, mask_imgs, targets

class ImageDataset(Dataset):
    """ Image dataset no bounding boxes """
    def __init__(self, image_file_path, use_pad=True, img_size=416):
        """
        image_file_path: path to text file with list of image file paths
        transform []: list of transforms to perform (padding done later)
        use_pad: Boolean default True, to use square padding
        img_size: resizing size for batches
        """
        self.image_file_path = image_file_path
        self.use_pad = use_pad      # boolean for using pad
        self.img_size = img_size    # resized img size

        with open(self.image_file_path, "r") as file:
            self.img_files = file.readlines()

    def __getitem__(self, index):
        """
        get an image
        """
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # convert to RGB file to PIL image to numpy array
        img = np.array(Image.open(img_path).convert("RGB"))
        height, width, channel = img.shape
        pad_size = max(width, height)

        # resize images to a specific size for batch stacking
        if self.use_pad:
            box_transforms = [iaa.PadToFixedSize(width=pad_size, height=pad_size,
                              position="center", pad_cval=0)]
            box_transforms += [iaa.Resize(self.img_size)]
        else:
            box_transforms = [iaa.Resize(self.img_size)]

        seq = iaa.Sequential(box_transforms)
        img_aug = seq(image=img)

        # convert to tensor
        img_aug = transforms.ToTensor()(img_aug)

        return img_path, img_aug

    def __len__(self):
        return len(self.img_files)