---
layout: distill
title: "YOLOv8: The Missing Paper"
description: A top down approach to building an object detector
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Anonymous

bibliography: 2025-04-28-yolov8-the-missing-paper.bib

toc:
  - name: HISTORY OF YOLO
  - name: 100FT VIEW OF OBJECT DETECTORS
  - name: NECK STRUCTURES
  - name: PREDICTION
  - name: IoU LOSS
  - name: DFL
  - name: BCE LOSS
  - name: LABEL ASSIGNMENT
  - name: THE C2f LAYER
  - name: YOLOv8 ARCHITECTURE
  - name: TRAINING
  - name: DATA AUGMENTATION

_styles: >
  .max100 {
    max-height: 100%;
  }
  .asdf {
    max-height: 75%;
  }
  .txyz {
    max-width: 75%;
  }
  .min {
    max-width: 30%;
  }

  .zebra {
    table, th, td {
      border: 3px solid white;
      border-collapse: collapse;
      color: #004C99;
    };
    tr:nth-child(odd) {
      background-color: #FCF0EF;
    }
    tr:nth-child(even) {
      background-color: #F8CECC;
    }
    th {
      background-color: #F8CECC;
      font-weight: bold;
      font-style: italic;
    }
    width: 120%;
  }

---


## History of YOLO

The history of YOLO is quite long and drama is also not missing. The first three versions <d-cite key="yolov1"></d-cite>, <d-cite key="yolov2"></d-cite>, <d-cite key="yolov3"></d-cite> are published by Joseph Redmon and are developed and trained using the Darknet framework<d-cite key="DarkNet"></d-cite>. YOLOv3<d-cite key="yolov3"></d-cite>, published in 2018, is the last version published by the original author, and it is not even an official paper, but rather a (very informal) technical report.

<div class="l-page-outset">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/yolo-history.svg" class="img-fluid asdf" %}
</div>

In 2020 Redmon tweeted in a post that he had stopped doing computer vision research, thus leaving the algorithm and its further development to the CV community:

{% twitter https://x.com/pjreddie/status/1230524770350817280 %}

And this is where things start to go crazy..

The two most active contributors to the development of YOLO are Glenn Jocher from Ultralytics and the research group of Chien-Yao Wang and Hong-Yuan Mark Liao joined by Alexey Bochovskiy. Glenn Jocher is responsible for the translation of YOLOv3 from Darknet into PyTorch, as well as for the models YOLOv5 and YOLOv8. While Alexey Bochovskiy, Chien-Yao Wang and Hong-Yuan Mark Liao are authors of YOLOv4<d-cite key="yolov4"></d-cite>, Scaled-YOLOv4<d-cite key="scaled-yolov4"></d-cite> and YOLOv7<d-cite key="yolov7"></d-cite>. (YOLOR<d-cite key="yolor"></d-cite> and YOLOv9<d-cite key="yolov9"></d-cite> are by the same research group without Alexey Bochovskiy.)

YOLOv4<d-cite key="yolov4"></d-cite> is still considered somewhat an "official" version mainly because of Alexey Bochovskiy. In another twitter post by Redmon he is praised for his contributions to developing the Darknet framework and YOLO:

{% twitter https://x.com/pjreddie/status/1253891078182199296 %}

YOLOv5, on the other hand, sparked a lot of heated debate, and already [the second issue opened on the YOLOv5 repository](https://github.com/ultralytics/yolov5/issues/2) in GitHub asks for a name change.

Nevertheless, being open-sourced, the models developed by Ultralytics are widely used and have a large base of contributors. These models are also designed to be modular and easy to train on a variety of datasets. And while newer YOLO models introduce ever more quirks and kinks to their architectures, YOLOv5 and YOLOv8 remain relatively simple and resemble closely the architecture of YOLOv4. However, as the focus is more on development and improvement, these models lack official publications and the only way to understand them is to read the source code.

In this post we will try to explain in detail the architecture of YOLOv8, discuss some of the design choices made, and layout the training procedure for the neural network. We hope that this will help other people to better understand not only this model, but also the general ideas behind the models from the YOLO family, as well as other single-shot detectors.

## 100ft View of Object Detectors

The architecture of the neural network for a single-shot object detector can be broken down into three components: *backbone*, *neck*, and *head*. The network is typically divided into *stages* or *levels*. On a given level $$ i $$ the computations are performed on a fixed resolution $$ H_i \times W_i $$ that has a *stride* $$ s $$ with respect to the initial image resolution, i.e. $$ H_i = H // s $$ and $$ W_i = W // s $$.

The output of the last layer on level $$ i $$ is denoted as the *feature map* $$ F_i $$, corresponding to the given resolution. This choice is natural, since we expect the deepest layer of each level to have the strongest features. This feature map is then fed into the head, which produces bounding box predictions for the given resolution.

{% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/object-detector.svg" class="img-fluid txyz" %}

In order to understand why object detectors are designed the way they are and why they differ from classification models, for example, we first need to discuss the challenges that object detection models face.

First of all, object detection models have a variable number of outputs. A given image might contain $$ 1 $$ or $$ 2 $$ objects that need to be detected, or it might contain $$ 1000 $$ objects. This is in stark contrast to oder tasks (e.g. classification, localization, segmentation, pose estimation, etc.), where the exact size of the output is known ahead of time. To address this issue the image is divided into grid cells and for every grid cell the model predicts a fixed number of bounding boxes together with a classification label for each box. Most of these predicted bounding boxes will be labelled as *background*, meaning there is no object there. And only those that have meaningful labels will be returned as the final output of the model.

A feature map $$ F_i $$ can be considered as imposing a grid of size $$ H_i \times W_i $$ on top of the input image. Lower resolution feature maps deeper in the network yield coarser grids, and higher resolution maps yield finer grids. The prediction head that processes this feature map will then output bounding box predictions ($$ 4 $$ numbers) for each of the pixels in the map, i.e. for each cell of the grid imposed by the map.

<div class="l-body-outset">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/grid-cell.svg" class="img-fluid asdf" %}
</div>

## Neck Structures

Another issue that arises with object detectors, and is not present in classification models, is that the model needs to be able to detect objects of different sizes on the same image. YOLOv1<d-cite key="yolov1"></d-cite> and YOLOv2<d-cite key="yolov2"></d-cite> models are missing the neck structure altogether and make bounding box predictions only at the final, lowest resolution feature map. These models suffer from low recall, and in particular they have issues detecting small and elongated objects. To see why this is the case note that a coarse-grained grid would be better for detecting larger objects on the image, while a fine-grained grid would be better for detecting smaller objects.

A first approach to solve this issue would be to directly make predictions from the feature maps at different stages of the backbone. This, in fact, is the approach used by the SSD model<d-cite key="SSD"></d-cite>. This, however, is sub-optimal as at the lowest resolution the input has been processed by all of the layers of the backbone network, producing semantically strong features. Whereas, at the highest resolution the input has been processed by very few layers, producing semantically weak features having low capacity for detection.

A better idea would be to add a top-down path, that would aggregate features across different levels. Feature maps from higher stages are upsampled and then merged with feature maps from the lower stages via lateral connections, yielding semantically strong features at all resolutions. This architecture is very similar to the U-net<d-cite key="Unet"></d-cite> used for segmentation tasks. Given that segmentation faces a similar problem of recognizing objects of different sizes on the same image, it is natural that segmentation and detection models would use similar architectures. This U-net-like architecture is called a feature pyramid network (FPN<d-cite key="FPN"></d-cite>) and is used in the YOLOv3<d-cite key="yolov3"></d-cite> model, as well as in other single-shot detectors like FCOS<d-cite key="FCOS"></d-cite> and RetinaNet<d-cite key="Focal"></d-cite>. According to <d-cite key="FPN"></d-cite> the addition of a top-down path improves mAP by $$ 2.0-2.3 $$, but most importantly, it significantly improves the recall of the model.

<div class="l-page-outset">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/feature-pyramid.svg" class="img-fluid" %}
</div>

Although the high resolution feature maps have semantically weaker features, their activations are more accurately localized as they were subsampled fewer times. This observation leads to the idea to construct an additional bottom-up path where feature maps from lower levels of the top-down path are downsampled and then merged with the feature maps from the upper levels, thus further improving the localization capacity of low-resolution maps. The combination of a top-down and a bottom-up path is called a path-aggregation network (PANet<d-cite key="PANet"></d-cite>) and starting with YOLOv4<d-cite key="yolov4"></d-cite> all subsequent YOLO models use a PANet neck. According to <d-cite key="PANet"></d-cite> the addition of a bottom-up path consistently improves mAP by more than $$ 0.9 $$.

YOLO models make use of one more additional layer that is added right after the final stage of the backbone, called a *Spatial Pyramid Pooling (SPP)* layer<d-cite key="SPP"></d-cite>. In its original version this layer applies a fixed number of **strided** pooling layers with **variable** kernel sizes, e.g. $$ [H \times W, H//2 \times W//2, H//4 \times W//4] $$. The kernel sizes are calculated at runtime depending on the input image size, and the produced outputs, resembling the layers of a pyramid, are flattened and concatenated to produce a fixed-length output.

The motivation for proposing this layer was to address the issue of non-fully convolutional networks requiring fixed size images as inputs (e.g. $$ 224 \times 224 $$). YOLO, being fully-convolutional, uses a modified version of *SPP* that has a fixed number of **padded, non-strided** *Max pooling* layers with **fixed** kernel sizes, e.g. $$ [5 \times 5, 9 \times 9, 13 \times 13] $$. The outputs of the pooling layers, having the same resolution, are then concatenated along the channel dimension together with the input. This increase in the receptive field of the final feature map reportedly improves mAP by around $$ 2.5 $$. <d-cite key="yolov4"></d-cite>

<dev class="l-body">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/spp.svg" class="img-fluid asdf" %}
</dev>

## Prediction

Finally, the feature map from every level of the bottom-up path is forwarded through a small CNN that is called a *head*. This fully convolutional network preserves the spatial dimensions and for every pixel of the feature map outputs predictions for bounding box coordinates and sizes. This type of prediction is known as *anchor-free prediction* and is the method used in YOLOv8. Most of the older YOLO models use *anchor-based prediction* because it yields much better recall. However, the accuracy of these models largely depends on the quality of the selected anchor box sizes. With the introduction of neck structures into the architecture and specialized loss functions for training, the use of anchor boxes is largely obsolete. Both YOLOX<d-cite key="yolox"></d-cite> and YOLOv6<d-cite key="yolov6"></d-cite> report mAP improvements of around $$ 1. $$ when switching from anchor-based to anchor-free prediction.

Note that in addition to detecting objects on the image, the model also needs to classify those objects into categories. Thus, together with the bounding box predictions, the model also outputs classification scores for every pixel. Optimizing for both regression and classification, however, introduces a conflict as reported in <d-cite key="RethinkingHead"></d-cite>, <d-cite key="RevisitingHead"></d-cite>. For this reason, object detectors tend to use a so called *decoupled head* with two separate branches for each of the tasks. The classification branch of the head predicts class scores for each pixel of the feature map, thus outputting a tensor of size $$ K \times H_i \times W_i $$, where $$ K $$ is the number of classes. The regression branch predicts the bounding box coordinate for each pixel of the feature map, outputting a tensor of size $$ 4 \times H_i \times W_i $$. YOLOX<d-cite key="yolox"></d-cite> was the first YOLO model to introduce head decoupling, followed by YOLOv6<d-cite key="yolov6"></d-cite> and YOLOv8. According to <d-cite key="yolox"></d-cite> the use of a decoupled head improves mAP by $$ 1.1 $$, while <d-cite key="yolov6"></d-cite> reports a mAP improvement of $$ 1.4 $$.

The weights of the head are shared across all feature map resolutions. This approach resembles traditional pyramid methods in image processing <d-cite key="ImagePyramid"></d-cite>, where a single classifier is applied across different scales and locations of the image.

{% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/decoupled-head.svg" class="img-fluid asdf" %}

For each bounding box prediction the model outputs $$ 4 $$ numbers. The three most common formats for describing the bounding boxes are:
 * $$ xywh $$ - center coordinates, width and height,
 * $$ xyxy $$ - coordinates of two opposite corners,
 * $$ ltrb $$ - distances to the four sides of the box measured from the grid cell center (also called the *anchor point* of the grid cell).

For prediction YOLOv8 adopts the $$ ltrb $$ bounding box format as proposed in <d-cite key="FCOS"></d-cite>. For every pixel of the feature map $$ F_i $$ the model predicts the distances $$ l, t, r, b, $$ to the four sides of the bounding box, measured from the anchor point corresponding to that pixel.

Given a feature map $$ F_i $$ of resolution $$ H_i \times W_i $$, let $$ s $$ be the stride of that map with respect to the input image. Every pixel from the feature map is mapped to the anchor point of its corresponding grid cell (i.e. the center of the cell). Thus, a pixel at location $$ (x,y) \in F_i $$ is mapped to the anchor point at location $$ (\frac{s}{2}+xs, \frac{s}{2}+ys) $$ on the input image. The predicted bounding box is then arrived at by measuring the distances $$ l, t, r, b, $$ from this anchor point.

<dev class="l-body">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/anchor-points.svg" class="img-fluid asdf" %}
</dev>

## IoU Loss

Given a bounding box prediction (i.e. $$ xyxy_\text{pred}$$) and its corresponding ground-truth bounding box (i.e. $$ xyxy_\text{gt}$$), a loss metric needs to be defined in order to optimize the model.

Early YOLO models used a simple mean-squared error (MSE) to directly regress on the bounding box coordinates and sizes. However, calculating the loss in this manner treats these values as independent variables, whereas they should be jointly regressed as a whole unit. Recognizing, this issue recent object detection models are instead optimized by calculating the loss as a function of the intersection over union (IoU) between predicted and ground truth box.

The standard IoU-loss is given by: $$ L_\text{IoU} = 1 - IoU $$. Other IoU losses augment the standard loss with a measure for central point distance (DIoU<d-cite key="DIoU"></d-cite>) and a measure for the aspect ratio (CIoU<d-cite key="DIoU"></d-cite>). Essentially, these add MSE of the center coordinates and MSE of the aspect ratio angle:

$$ L_{DIoU} = 1 - IoU + \dfrac{\text{dist}^2(C^P, C^{GT})}{D^2}, $$

$$ L_{CIoU} = 1 - IoU + \dfrac{\text{dist}^2(C^P, C^{GT})}{D^2} + \alpha V, $$

where $$ C^P $$ and $$ C^{GT} $$ are the center points of the predicted and the ground truth boxes, $$ D $$ is the diagonal length of the smallest enclosing box covering the two boxes, and $$ V $$ and $$ \alpha $$ are given as:

$$ V = \dfrac{4}{\pi^2} \bigg( \arctan\dfrac{w^{GT}}{h^{GT}} - \arctan\dfrac{w}{h} \bigg) ^2, $$

$$ \alpha = \dfrac{V}{1 - IoU + V}. $$

Using an IoU-based loss function reportedly improves mAP by $$ 1.0-2.0 $$. <d-cite key="DIoU"></d-cite>

 * The loss function used in YOLOv8 is CIoU loss: [`ultralytics/utils/loss.py#L102`](https://github.com/ultralytics/ultralytics/blob/b7e9e91d4655b568f6244494be53a391c51c7ffc/ultralytics/utils/loss.py#L102)

<div class="l-body">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/bbox-iou.svg" class="img-fluid asdf" %}
</div>

## DFL

In order to obtain more reliable and accurate bounding box predictions it is suggested that, instead of directly predicting the box dimensions, the network should predict a probability distribution over a range of values. That is, given a minimum value $$ v_\min $$ and a maximum value $$ v_\max $$ for a specific variable $$ v $$, for example the distance to the left side of the box $$ l $$, the interval $$ [v_\min, v_\max] $$ is discretized into a set of values $$ (v_0=v_\min, v_1, \dots, v_{n-1}, v_n=v_\max) $$ with equal spacing $$ \Delta $$. The model then predicts a probability distribution over this set and the final predicted value is calculated by taking the expectation:

$$ v_\text{pred} = \sum v_i p(v_i). $$

The model is optimized by maximizing the likelihood of the ground-truth value $$ \hat{v} $$ under the modelled distribution. However, since this value is most likely not present in the set of discrete values, its mass will be proportionally distributed between the two closest values $$ v_i $$ and $$ v_{i+1} $$ $$( v_i \leq \hat{v} \leq v_{i+1} )$$. The loss function, called *Distribution Focal Loss* (DFL)<d-cite key="DFL"></d-cite> is given by:

$$ L_\text{DFL} = -(v_{i+1}-\hat{v})\log p(v_i) -(\hat{v}-v_i)\log p(v_{i+1}). $$

With the use of *DFL*, the output of the regression branch of the head becomes instead a tensor of size $$ 4D \times H_i \times W_i $$, where $$ D = n+1 $$ is the number of discrete values in the interval $$ (v_\min, v_\max) $$. The scores predicted by the regression branch are then run through a softmax layer to arrive at the probabilities for the discrete values $$ p(v_0), p(v_1), \dots, p(v_n) $$.

What is left is for the predicted variables, $$ l, t, r, b, $$ to define the minimum and maximum possible values that they can attain, as well as the discretization spacing $$ \Delta $$. The choice of $$ v_\min $$ and $$ v_\max $$, directly influences how large and how small the predicted boxes can be. Assuming that the detection model will be used on images no larger than $$ 1024 \times 1024 $$, then a natural choice would be $$ v_\min=0, v_\max \approx 500 $$. The choice of $$ \Delta $$, however, is a bit more difficult. Choosing $$ \Delta $$ to be too large would affect the accuracy of the model, especially when predicting smaller boxes, as the model needs to learn very accurate distributions over the small set of discrete values. Choosing $$ \Delta $$ to be too small, on the other hand, would slow down, or even completely overwhelm the training, as for a lot of the discrete values there will be no samples in the training set. Ideally, what we would like, is to have larger $$ v_\max $$ and $$ \Delta $$ values when making predictions on the low-resolution feature maps where we expect to predict larger boxes, and smaller $$ v_\max $$ and $$ \Delta $$ values when making predictions on the high-resolution feature maps where we expect to predict smaller boxes.

In order to address this issue, when making predictions on a feature map $$ F_i $$, YOLOv8 predicts the variables $$ l, t, r, b, $$ on the resolution of that map, instead of on the resolution of the input image. Note that in doing so we need to scale them afterwards with the stride $$ s $$ of that feature map. Thus, we are effectively using $$ sv_\min, sv_\max, s\Delta $$ as parameters, meaning that for every resolution we are making use of different values.

The lowest resolution feature map of YOLOv8 has a stride $$ s=2^5=32 $$. Thus, for the model to have the capacity to predict bounding boxes of size $$ \approx 1000 \times 1000 $$ we would need to pick $$ v_\max \approx \frac{1000}{2s} = \frac{1000}{64} \approx 15 $$. Choosing $$ \Delta=1 $$ seems natural at this point, as it results in the discrete values from the interval being split precisely along the pixel locations on every feature map.

Choosing $$ v_\max=15, \Delta=1 $$ is also in agreement with the proposed values in the original paper <d-cite key="DFL"></d-cite>, where the authors report an improvement of around $$ 0.5 $$ mAP when using DFL.

 * YOLOv8 uses DFL loss: [`ultralytics/utils/loss.py#L97`](https://github.com/ultralytics/ultralytics/blob/b7e9e91d4655b568f6244494be53a391c51c7ffc/ultralytics/utils/loss.py#L97)
 * YOLOv8 uses $$ v_\max=15 $$: [`ultralytics/nn/modules/head.py#L39`](https://github.com/ultralytics/ultralytics/blob/b7e9e91d4655b568f6244494be53a391c51c7ffc/ultralytics/nn/modules/head.py#L39)
 * YOLOv8 uses $$ \Delta=1 $$: [`ultralytics/nn/modules/block.py#L66`](https://github.com/ultralytics/ultralytics/blob/77c3c0aaac25e8738c0fe976f3e17c65aca12445/ultralytics/nn/modules/block.py#L66)

<div class="l-page-outset">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/dfl.svg" class="img-fluid max100" %}
</div>

<!--
*Sidenote: since YOLOv8 makes the predictions in the resolution of the feature map, when computing the IoU loss instead of scaling the predictions up, they actually scale the ground-truth boxes down. Obviously, IoU being scale invariant, it doesn't matter if you scale up or down. However, when scaling up the predictions, we also shift them with \\(s/2\\) and this shifting is not accounted for in the code when scaling the ground-truth boxes down.*

https://github.com/ultralytics/ultralytics/blob/b7e9e91d4655b568f6244494be53a391c51c7ffc/ultralytics/utils/loss.py#L252
-->

## BCE Loss

In addition to optimizing the bounding box predictions, the model is also optimized for predicting the correct class of the detected objects. Instead of using softmax multi-class classification, YOLOv8 (and most of the other YOLO models) perform multi-label classification and are optimized by training independent binary classifiers for each class. Given the classification scores $$ z $$ predicted by the model for a given pixel from the feature map $$ F_i $$, and the ground-truth class, the binary-cross entropy loss is computed as:

$$ L_\text{BCE} = \sum_{c=1}^K \Big( y_c \log \sigma(z_c) + (1 - y_c)\log(1-\sigma(z_c)) \Big), $$

where $$ y_c = 1 $$ if $$ c $$ is the ground truth class, otherwise $$ y_c = 0 $$.

The classification loss $$ L_\text{BCE} $$ is calculated separately for the prediction at every pixel of a given feature map $$ F_i $$. Predictions that were not assigned a ground-truth label during the label assignment procedure (discussed next), the so called *negative examples*, will have $$ y_c = 0 \quad \forall \quad 1 \leq c \leq K $$. Since most of the predictions could actually be negative examples, there is a concern that this imbalance in the training data leads to inefficient training and might even overwhelm the training and lead to degenerate models. To address this issue the Focal Loss<d-cite key="Focal"></d-cite> was proposed, which includes an additional scaling factor that automatically down-weights the contribution of training examples that are correctly classified with high probability, the so-called *easy examples*:

$$ L_\text{focal} = \sum_{c=1}^K \Big( y_c \boldsymbol{(1-\sigma(z_c))^\gamma} \log \sigma(z_c) + (1 - y_c) \boldsymbol{\sigma(z_c)^\gamma} \log(1-\sigma(z_c)) \Big). $$

The VeriFocal loss<d-cite key="VeriFocal"></d-cite> further extends this equation by making the scaling factor a function of the IoU between the predicted box and the ground-truth box. In the codebase of YOLOv8, code for Focal Loss and VeriFocal Loss is included, but is not used.

 * YOLOv8 uses binary cross-entropy loss as its default classification loss function: [`ultralytics/utils/loss.py#L166`](https://github.com/ultralytics/ultralytics/blob/e48a42ec5dbbed3b3a8a95e6ab10ab0eeea4e518/ultralytics/utils/loss.py#L166)
 * It was reported that YOLOv8 no longer supports the implementation of VeriFocal loss because it did not show any improvements in their experiments: [`ultralytics/issues/1448`](https://github.com/ultralytics/ultralytics/issues/1448#issuecomment-1471570296)

The total loss is computed as a weighted sum of the three losses $$ L_\text{BCE}, L_\text{DFL} $$ and $$ L_\text{CIoU} $$:

$$ L = \frac{1}{N_\text{pos}} \sum \bigg( \lambda_0 L_\text{BCE} + \mathbb{1}^\text{pos} \lambda_1 L_\text{DFL} + \mathbb{1}^\text{pos} \lambda_2 L_\text{CIoU} \bigg), $$

where the summation is over all predictions across all feature map resolutions. The indicator function $$ \mathbb{1}^\text{pos} $$ is $$ 1 $$ if this prediction is a positive example and $$ 0 $$ otherwise, and $$ N_\text{pos} $$ is the number of positive examples from this summation. The weights $$ \lambda_0, \lambda_1 $$ and $$ \lambda_2 $$ are hyperparameters to balance the contribution of the different losses.

 * YOLOv8 uses $$ \lambda_0=\lambda_\text{cls}=0.5, \lambda_1=\lambda_\text{dfl}=1.5, \lambda_2=\lambda_\text{box}=7.5 $$: [`ultralytics/cfg/default.yaml#L97-L99`](https://github.com/ultralytics/ultralytics/blob/e72f19f5cfd5fe7bb8b09a156933e96bd90581bf/ultralytics/cfg/default.yaml#L97-L99)

<!--
Add an image showing a breakdown of the loss function.
Get motivation from YOLOv1.
-->

## Label Assignment

The training data for object detection models consists of pairs $$ (x_i, y_i) $$, where $$ x_i $$ gives the raw pixel values of an image, and $$ y_i $$ is a list of the ground-truth bounding box coordinates (in $$ xyxy $$ format in the case of MS-COCO) together with the class of the object surrounded by the corresponding box. The output of the network, on the other hand, consists of bounding box predictions and class predictions for the grid cells on several resolution levels. This mismatch between the number of predictions and the number of ground-truth labels necessitates the use of some procedure for matching labels to predictions, i.e. assigning a label to every prediction.

YOLOv1<d-cite key="yolov1"></d-cite> uses a simple "center-based" approach for label assignment that, for a given ground-truth bounding box, finds which grid cell contains the center of that box. Then the prediction for that cell is labeled with the given ground-truth bounding box. Grid cells that do not contain any central point are then considered as negative examples for the model. FCOS<d-cite key="FCOS"></d-cite> extends this idea and, instead of the center point, finds all grid cells that contain *any point* of a given ground-truth bounding box. Predictions for these cells are then labeled with the given ground-truth bounding box only if the predicted class is the same as the ground-truth. Obviously, FCOS suffers from too many positive examples, while YOLOv1 has too few.

A more elaborate procedure for label assignment, known as Task Aligned Assignment (TAL) and proposed in <d-cite key="TOOD"></d-cite>, defines a *proximity metric* between a predicted bounded box and a ground-truth bounding box as:

$$ t = s^\alpha \times u^\beta, $$

where $$ s $$ is the predicted probability for the ground truth class, $$ u $$ is the IoU between the two boxes, and $$ \alpha $$ and $$ \beta $$ are hyperparameters to control the contribution of each of the factors.

For every ground-truth bounding box we compute its proximity metric with all of the predictions, and assign it as label to the top $$ k$$  predictions.

The proposed metric actually looks like a generalized version of the procedure proposed in <d-cite key="FCOS"></d-cite>. The value of $$ u $$ tracks the amount of points from the ground truth bounding-box that are contained within a given grid cell. The value of $$ s $$ shows how well the prediction scores the ground-truth class. The value of $$ k $$ is used to prevent excessive numbers of positive examples.

One issue that arises with all of these assignment methods, however, is that a prediction might receive multiple ground-truth bounding boxes assigned as labels.

 * YOLOv8 solves this issue by simply assigning as label only the ground-truth box yielding the highest IoU with the prediction: [`ultralytics/utils/tal.py#L284-L292`](https://github.com/ultralytics/ultralytics/blob/3e48829cd628d1a5a8c8230787d691a971c33496/ultralytics/utils/tal.py#L284-L292)

The authors of YOLOX<d-cite key="yolox"></d-cite> were the first to bring up the problem of label assignment in the YOLO models, showing in their paper that using a dynamic strategy (in this case SimOTA) improves the mAP by $$ 2.3 $$. While the authors of YOLOv6<d-cite key="yolov6"></d-cite> were the first to run an ablation study on different label assignment methods showing that TAL<d-cite key="TOOD"></d-cite> yields the highest mAP improvement. Consequently TAL is the label assignment method adopted by the YOLOv8 model as well.

 * YOLOv8 uses TAL as label assignment method with hyperparameters $$ \alpha=0.5, \beta=6.0, k=10 $$: [`ultralytics/utils/loss.py#L176`](https://github.com/ultralytics/ultralytics/blob/e48a42ec5dbbed3b3a8a95e6ab10ab0eeea4e518/ultralytics/utils/loss.py#L176)

<!--
Add an image showing how TAL works.
![Label Assignment](img/label_assignment.png "Label Assignment")
-->

In addition to the label assignment procedure, the authors of TAL<d-cite key="TOOD"></d-cite> also propose to change the classification objective $$ L_\text{BCE} $$ of the positive examples. For a given positive example, i.e. prediction with assigned ground-truth bounding box, instead of computing the cross-entropy with the binary label $$ y_c $$, the IoU-normalized proximity metric is used:

$$ L_\text{BCE} = \sum_{c=1}^K \Big( y_c \log \sigma(z_c) + (1 - y_c)\log(1-\sigma(z_c)) \Big), $$

where:

$$ \boldsymbol{y_c = t \frac{\max_\text{gt}(u)}{\max_\text{gt}(t)}}, $$

if $$ c $$ is the ground truth class, otherwise $$ y_c=0 $$.

Here $$ t $$ is the proximity metric between the prediction and the ground-truth bounding box, and the $$ \max $$ operation is taken over all other predictions to which this ground-truth bounding box is assigned as label.

 * YOLOv8 also implements this modification to the BCE loss: [`ultralytics/utils/tal.py`](https://github.com/ultralytics/ultralytics/blob/3e48829cd628d1a5a8c8230787d691a971c33496/ultralytics/utils/tal.py#L112-L116)

## The *C2f* Layer

The main building block of YOLOv8 is the *C2f* convolution layer, which is a *CSPNet*-style block<d-cite key="CSPNet"></d-cite>. In the beginning of the CSPNet the input tensor is split in half along the channel dimension and only one half is forwarded through the layers of the network, while the other half is skip-connected to the end where the two halves are concatenated. The CPSNet architecture can be applied on top of any network. Its main purpose is *not* to improve the model accuracy, but rather to reduce the computational and memory costs while introducing only minor degradation to the accuracy.

The cross-stage partial connection was developed by trying to improve on the DenseNet<d-cite key="DenseNet"></d-cite>, as these networks introduce the largest amount of redundancy. But even the CSPDenseNets are still too demanding to be used for real-time object detection. A more efficient alternative to DenseNets are VoVNets<d-cite key="VoVNet"></d-cite>, which instead of concatenating features at every subsequent layer, proposes to concatenate features only once in the last feature map. This "*CPSVoVNet*" block is very similar to the *CSPOSANet* block used in the Scaled-YOLOv4-tiny<d-cite key="scaled-yolov4"></d-cite> model and to the *G-ELAN* block reinvented for the YOLOv9<d-cite key="yolov9"></d-cite> model.

The *C2f* layer used in YOLOv8 also has the architecture of *CSPVoVNet*. This layer consists of residual blocks, called a *Darknet bottleneck*, made of two $$ 3 \times 3 $$ convolution layers. The *Darknet bottleneck* residual blocks are connected as a *VoVNet* and finally a CSPNet architecture is applied on top.

 * Even though this block is called *Darknet* ***bottleneck***, it is applied without a bottleneck in the C2f layer of YOLOv8: [`ultralytics/nn/modules/block.py#L233`](https://github.com/ultralytics/ultralytics/blob/652e05e833087652053346946827034a4a038025/ultralytics/nn/modules/block.py#L233)

Batch normalization and SiLU<d-cite key="silu"></d-cite> non-linearity are applied after every convolution layer. This pattern is called a *CBS-block* and is used throughout the entire model.

<dev class="l-body-outset">
  {% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/C2f.svg" class="img-fluid asdf" %}
</dev>

## YOLOv8 Architecture

The backbone network of YOLOv8 consists of five stages, where each stage consists of a downsampling layer and a *C2f* block. Downsampling reduces the spatial dimensions in half and doubles the channels of the input feature map from the previous stage. The downsampling operation is performed using a convolution layer with kernel size $$ 3 \times 3 $$ and stride $$ s=2 $$. The very first stage of the backbone has only a downsampling layer, resulting in a more aggressive downsampling in the beginning.

The depth of the *C2f* block, i.e. the number of convolution layers within the block, varies with stages. A general rule for classification models is to increase the depth in each stage, but not necessarily the last stage. This design consideration, as well as many more, were analyzed in <d-cite key="RegNet"></d-cite>. It is expected that these findings hold true to some extent for detection models as well. YOLOv5 employs a ratio of 1:2:3:1, while in the newer YOLOv8 this is changed to 1:2:2:1.

After the last stage of the backbone a modified *Spatial Pyramid Pooling* layer is applied that uses three *Max Pooling* layers with kernel sizes $$ [5 \times 5, 9 \times 9, 13 \times 13] $$. The use of these exact kernel sizes for the *SPP* block is not an accident. Note that when using a $$ 5 \times 5 $$ kernel, the receptive field of a $$ 5 \times 5 $$ patch of pixels is $$ 9 \times 9 $$. Similarly, again using a $$ 5 \times 5 $$ kernel, the receptive field of a $$ 9 \times 9 $$ patch of pixels is $$ 13 \times 13 $$. Thus, an equivalent representation for the modified *SPP* layer is to apply three consecutive Max Pooling layers, each with kernel size $$ 5 \times 5 $$. In addition, $$ 1 \times 1 $$ convolutions are applied before and after the pooling layers. This equivalent representation together with the convolution layers is called *Spatial Pyramid Pooling Fast (SPPF)* layer.

 * YOLOv8 uses the first $$ 1 \times 1 $$ convolution in the *SPPF* layer as a bottleneck: [`ultralytics/nn/modules/block.py#L179-180`](https://github.com/ultralytics/ultralytics/blob/652e05e833087652053346946827034a4a038025/ultralytics/nn/modules/block.py#L179-L180)

The neck is a PANet consisting of three levels. In the top-down path upsampling is performed using the nearest neighbor strategy and after that the two maps are concatenated along the channel dimension. Due to the aliasing effect of the upsampling operation, the output needs to be forwarded through a convolution layer afterwards. The original FPN<d-cite key="FPN"></d-cite> paper proposes to use a single $$ 3 \times 3 $$ convolution, while YOLOv8 uses a *C2f* block instead. This choice is natural, as the *C2f* is likely more efficient than a standard convolution because of the cross-stage partial connection.

It is important to note that while the *C2f* blocks in the backbone use residual connections, this is not the case in the neck. Obviously, adding a residual connection would not help in reducing the aliasing effect of upsampling, and for that reason it is left out.

In the bottom-up path downsampling is performed using a $$ 3 \times 3 $$ convolution with a stride of $$ s=2 $$ and the two maps are again concatenated along the channel dimension. After merging the two maps, the output is again forwarded through a *C2f* block without residual connections.

The depth of the *C2f* blocks in the PANet is equal to the depth of the final stage of the backbone, while their width matches the width of the level at which they are applied.

Finally, each of the branches of the head consists of two *CBS* blocks followed by a $$ 1 \times 1 $$ convolution that scales the number of channels to match the expected number of outputs. The output channels of the first *CBS* block in the regression and the classification branches respectively, are calculated as:

$$ C_2 = \max(16, C_7 // 4, 4*D), $$

$$ C_3 = \max(C_7, \min(K, 100)), $$

where $C_7$ are the channels of the feature map with the highest resolution. [`ultralytics/nn/modules/head.py#L42`](https://github.com/ultralytics/ultralytics/blob/652e05e833087652053346946827034a4a038025/ultralytics/nn/modules/head.py#L42)


{% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/head.svg" class="img-fluid" %}


YOLOv8 comes in five different model sizes: *nano* (n), *small* (s), *medium* (m), *large* (l), and *extra-large* (x). All of these models have the same architecture and structure, and the only difference is in their depth and width.


<div class="l-body-outset table zebra">
<div markdown=1>

|     | Layers                     | nano        | small         | medium        | large         | extra         |
|-----|----------------------------|-------------|---------------|---------------|---------------|---------------|
| S1  | CBS 3x3, s=2               | C=16        | C=32          | C=48          | C=64          | C=80          |
| S2  | CBS 3x3, s=2 + C2f         | C=32,  N=1  | C=64,  N=1    | C=96,  N=2    | C=128,  N=3   | C=160, N=3    |
| S4  | CBS 3x3, s=2 + C2f         | C=64,  N=2  | C=128, N=2    | C=192, N=4    | C=256,  N=6   | C=320, N=6    |
| S4  | CBS 3x3, s=2 + C2f         | C=128, N=2  | C=256, N=2    | C=384, N=4    | C=512,  N=6   | C=640, N=6    |
| S5  | CBS 3x3, s=2 + C2f         | C=256, N=1  | C=512, N=1    | C=576, N=2    | C=512,  N=3   | C=640, N=6    |
|SPPF | Conv 1x1 + 3xMP + Conv 1x1 | C=128, 256  | C=256, 512    | C=288, 576    | C=256, 512    | C=320, 640    |
| P4  | Up + Concat  + C2f         | C=128, N=1  | C=256, N=1    | C=384, N=2    | C=512,  N=3   | C=640, N=3    |
| P3  | Up + Concat  + C2f         | C=64,  N=1  | C=128, N=1    | C=192, N=2    | C=256,  N=3   | C=320, N=3    |
| N4  | CBS 3x3, s=2 + Concat + C2f| C=128, N=1  | C=256, N=1    | C=384, N=2    | C=512,  N=3   | C=640, N=3    |
| N5  | CBS 3x3, s=2 + Concat + C2f| C=256, N=1  | C=512, N=1    | C=576, N=2    | C=512, N=3    | C=640, N=3    |
| Hcls| CBS 3x3 + CBS 3x3 + CBS1x1 | C=80, 80, 80| C=128, 128, 80| C=192, 192, 80| C=256, 256, 80| C=320, 320, 80|
| Hreg| CBS 3x3 + CBS 3x3 + CBS1x1 | C=64, 64, 64| C=64, 64, 64  | C=64, 64, 64  | C=64, 64, 64  | C=80, 80, 64  |

</div>
</div>


## Training

The training procedure of YOLOv8 is relatively simple and straight-forward. The model is trained [for $$ 500 $$ epochs](https://github.com/ultralytics/ultralytics/issues/852) on the MS-COCO detection dataset with a *Nesterov SGD* optimizer. The default batch size for training is $$ 16 $$. However, the model is trained using gradient accumulation, thus the gradients are backpropagated every $$ 4 $$ iterations. This results in a total of $$ \approx 230 000 $$ gradient update steps.

The model is trained from scratch without any pretraining on classification tasks. As reported in <d-cite key="PreTraining"></d-cite>, pre-training speeds up convergence early in the training, but does not necessarily yield the best final performance. Starting with Scaled-YOLOv4<d-cite key="scaled-yolov4"></d-cite> all YOLO models are trained from scratch.

The training process starts with a warm-up stage of 3 epochs. During that stage the gradient accumulation slowly kicks in; in the beginning gradients are backpropagated at every iteration and in the end of the warm-up stage - at every $$ 4 $$ iterations. During that stage also the learning rate increases linearly from $$ 0 $$ to the initial learning rate $$ \text{lr}_0 = 0.1 $$, and the moment increases linearly from $$ 0.8 $$ to $$ 0.9 $$. However, the learning rate of the biases is set to $$ 0.1 $$ from the beginning of the training and does not warm-up.

After every epoch the learning rate is decayed following a linear schedule from $$ \text{lr}_0 = 0.1 $$ in the beginning to $$ \text{lr}_0*\text{lr}_f = 0.1 * 0.1 = 0.01 $$ in the end of training. The learning rate at epoch $$ i $$ is given by:

$$ \text{lr}_i = ((1 - \frac{i}{N})(1 - \text{lr}_f) + \text{lr}_f) \text{lr}_0, $$

where $$ N $$ is the total number of epochs.

Before backpropagating the gradients, they are clipped by norm to a value of $$ 10 $$. In addition, weight decay is set to $$ 5e-4 $$, but is turned of for the normalization layers and the biases.

 * YOLOv8 does not use weight decay on the biases and on the normalization layers: [`ultralytics/engine/trainer.py#L792-L814`](https://github.com/ultralytics/ultralytics/blob/0183eaeeba746bb20a88a2b5e7af5e4078249bac/ultralytics/engine/trainer.py#L792-L814)

During the training process YOLOv8 maintains a second (shadow) copy of the model called the *EMA model*<d-cite key="ema"></d-cite>. At every iteration of the training procedure the parameters of the EMA model $$ \theta' $$ are updated completely independently as an exponential moving average of the original model parameters $$ \theta $$:

$$ \theta_t' = \alpha_t \theta_{t-1}' + (1-\alpha_t)\theta_{t}. $$

The smoothing coefficient $$ \alpha_t $$ is itself an exponentially decaying variable:

$$ \alpha_t = d(1-e^{-t / \tau}), $$

with default values $$ d=0.9999 $$ and  $$ \tau=2000 $$.

Unlike the procedure outlined in <d-cite key="ema"></d-cite>, in YOLOv8 the EMA model does not participate in the training of the original model. At the end of the training procedure the original model is discarded and only the EMA model is kept.

 * YOLOv8 uses the final EMA model for making inference: [`ultralytics/engine/trainer.py#L522`](https://github.com/ultralytics/ultralytics/blob/1c6bfd3039d0456c18611f475ace661111413c85/ultralytics/engine/trainer.py#L522)

Finally, an early stopping procedure is adopted, that at the end of every epoch, tests the model on the validation set and records the achieved mAP. Training stops if no improvement was seen in the last $$ 100 $$ epochs. The mAP on the validation set is calculated using the EMA Model.

On the MS-COCO validation dataset YOLOv8 manages to achive some improvement over the previous models from the YOLO family both in terms of accuracy and in terms of speed. However, arguably the main contribution is the clean approach that YOLOv8 adopts.

{% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/yolo-results.svg" class="img-fluid" %}

## Data Augmentation

The model is trained on input images of size $$ 640 \times 640 $$. The following data augmentation methods are applied in this exact order when training YOLOv8:
 * [`ultralytics/data/augment.py#L2330-L2335`](https://github.com/ultralytics/ultralytics/blob/3e48829cd628d1a5a8c8230787d691a971c33496/ultralytics/data/augment.py#L2330-L2335)

{% include figure.html path="assets/img/2025-04-28-yolov8-the-missing-paper/data-augmentation.svg" class="img-fluid min" %}

Mosaic augmentation is applied with a probability of $$ 1.0 $$, but is closed in the final $$ 10 $$ epochs as proposed in <d-cite key="yolox"></d-cite>. During mosaic augmentation, four random images are combined together by placing them on a $$ 2 \times 2 $$ grid, resulting in a $$ 1280 \times 1280 $$ image. The point of contact is also chosen at random somewhere in the mid $$ 320 \times 320 $$ section of the image. If any of the combined images is out of bounds it gets cropped. The empty portion of the image is filled with gray pixels with value $$ 114 $$:

 * [`ultralytics/data/augment.py#L657-L713`](https://github.com/ultralytics/ultralytics/blob/3e48829cd628d1a5a8c8230787d691a971c33496/ultralytics/data/augment.py#L657-L713)

After the mosaic augmentation, affine transformations are applied that scale the image by a random factor in the interval $$ (0.5; 1.5) $$ and translate it with a random number of pixels in the interval $$ (-0.1, 0.1) * 1280 $$ in both x and y directions.

The input is further augmented with:
 * Albumentations - Uniform Blur, Median Blur, To Grayscale and CLAHE all with probability $$ p = 0.01 $$;
 * Random HSV - a random value in $$ [-0.015, 0.015] $$ is added to Hue; a random value in $$ [-0.7, 0.7] $$ is added to Saturation; a random value in $$ [-0.4, 0.4] $$ is added to Value.
 * Random Flip - finally the image is flipped horizontally with probability $$ p = 0.5 $$. Images are not flipped vertically.

Mosaic augmentation and affine transformations are known as *strong augmentation* while the other transformations Albumentations, Random HVS and Random Flip, are known as *weak augmentation*.


