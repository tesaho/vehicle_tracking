# pytorch_yolov3
This is a implementation of YOLOv3 based off of Erik Lindernoren's [PyTorch_YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3).

For vehicle detection, the training set was composed of images from Stanford's Cars dataset and combined with hand relabelled
images from NEXET cars dataset.  

## Installation
##### Clone and install requirements
    $ git clone https://github.com/tesaho/pytorch_yolov3
    $ cd pytorch_yolov3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Unzip cars.zip
    $ cd data/
    $ unzip cars.zip .

## Train
Train a model and save in a folder MODEL_NAME. Additional features:

- re-start at a specific checkpoint RESTART_POINT.
- hyperparameter optimization (CONF_THRES, NMS_THRES, IOU_THRES, LEARNING_RATE)

Outputs the following in MODEL_NAME/:

- checkpoints/ (model weights)
- logs/ (tensorboard log files)
- outputs/parameters.txt
- outputs/valiation_maps.csv

```
$ python3 train.py [-h] [--model_name MODEL_NAME]
                [--epochs EPOCHS] 
                [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] 
                [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] 
                [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--restart_point RESTART_POINT]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
                [--conf_thres CONF_THRES]
                [--nms_thres NMS_THRES]
                [--iou_thres IOU_THRES]
                [--optimizer OPTIMIZER]
                [--learning_rate LEARNING_RATE]
                
```

#### Train example
To train on cars data set using a Darknet-53 backend pretrained on ImageNet run: 
```
$ python3 train.py --model_name darknet_cars --epochs=10 --data_config config/cars_small.data  --pretrained_weights weights/darknet53.conv.74
```

#### Training log
```
---- [Epoch 2/2, Batch 0/2] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 11           | 22           | 44           |
| loss       | 78.568298    | 75.857491    | 79.995689    |
| x          | 0.044378     | 0.086360     | 0.068668     |
| y          | 0.018975     | 0.152630     | 0.099213     |
| w          | 0.363917     | 1.900934     | 4.415801     |
| h          | 0.330206     | 0.037664     | 4.148860     |
| conf       | 77.089592    | 73.024803    | 70.526222    |
| cls        | 0.721232     | 0.655100     | 0.736922     |
| cls_acc    | 0.00%        | 0.00%        | 0.00%        |
| recall50   | 0.000000     | 0.000000     | 0.000000     |
| recall75   | 0.000000     | 0.000000     | 0.000000     |
| precision  | 0.000000     | 0.000000     | 0.000000     |
| conf_obj   | 0.485040     | 0.479717     | 0.502489     |
| conf_noobj | 0.524801     | 0.509523     | 0.499409     |
+------------+--------------+--------------+--------------+
Total loss 234.42147827148438
---- ETA 0:00:00.551487
```

## Test

Outputs in MODEL_NAME/eval/

- parameters.txt
- PRETRAINED_WEIGHTS_batch_results_iou_IOU_THRES.csv
- PRETRAINED_WEIGHTS_map_iou_IOU_THRES.csv
- PRETRAINED_WEIGHTS_predictions_iou_IOU_THRES.csv

```
$ python3 test.py [-h] [--model_name MODEL_NAME]
                [--batch_size BATCH_SIZE]
                [--model_def MODEL_DEF] 
                [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] 
                [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--conf_thres CONF_THRES]
                [--nms_thres NMS_THRES]
                [--iou_thres IOU_THRES]
```

#### Test example
To test on cars validation set using our previous Darknet53 model.
```
$ python3 test.py --model_name darknet_cars --data_config config/cars_small.data  --pretrained_weights weights/darknet53.conv.74
```

## Detections
Uses pretrained weights to draw bounding boxes on images and make predictions. 

Outputs in MODEL_NAME/:

- detections (predictions per image)
- img_boxes (image with bounding boxes)


```
$ python3 test.py [-h] [--model_name MODEL_NAME]
                [--image_path IMAGE_PATH]
                [--batch_size BATCH_SIZE]
                [--model_def MODEL_DEF] 
                [--pretrained_weights PRETRAINED_WEIGHTS] 
                [--class_Path CLASS_PATH]
                [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--conf_thres CONF_THRES]
                [--nms_thres NMS_THRES]
                [--iou_thres IOU_THRES]
                [--checkpoint_model CHECKPOINT_MODEL]
```
#### Detect example 
To produce predictions and bounding boxes on cars validation images.
```
$ python3 detect.py --model_name darknet_cars --checkpoint_model darknet_cars/checkpoint/darknet_cars_checkpoint_10
```

## Tensorboard
Track training progress in Tensorboard:

* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```

## Train on Custom Dataset

#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.  The last row should be empty.

```
sedan
hatchback
bus
pickup
minibus
van
truck
motorcycle
suv

```

#### Image Folder
Move the images of your dataset to `data/custom/train_images/` and `data/custom/valid_images`.

#### Annotation Folder
Move your annotations to `data/custom/train_labels/` and `data/custom/valid_labels/`. 
The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

Example: nexet_000003.txt
```
0.0 0.4527777777777344 0.4762845849798611 0.07222222222265628 0.09486166007916666
5.0 0.10833333333320312 0.5217391304354166 0.13666666666640626 0.20158102766805558
```

#### Define Train and Validation Sets
List the paths to all the images in the train_images folder to `data/custom/train.txt` and the paths to all the images 
in valid_images folder to `data/custom/valid.txt`.  

To create a `train.txt`, run the commands:

```
$ cd data/custom/train_images/
$ printf '%s\n' "$PWD"/* > train.txt
$ mv train.txt ../.
```

## Credit

Original Darknet53 model from below.

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```


