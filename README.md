# RandlyWiredNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)

This is an Unofficial implementation of: [Randomly Wired Neural Network](https://arxiv.org/abs/1904.01569)(In Progress).
![](misc./network.png)

### Progress

* Currently training on VOC dataset with ~50% mean Accuracy Precision score.
  * Trained on lr = 0.05, epoch = 50, small regime, C = 78, graph type = WS, k = 4, p = 0.75

### Project Overview

In this project, I am going to implement one of the image classification model generated using one of the state-of-the-art NAS method.

##### NAS (Computer Vision)

NAS, Neural Architecture Search, is essentilly an algorithm that allows auomatic buildilng of a neural network model given parameters. Unlike hand-designed NN architectures such as [DenseNet](https://arxiv.org/pdf/1608.06993.pdf), [ResNet](https://arxiv.org/pdf/1512.03385.pdf), etc., an NAS lets the heuristics design how to information is processed and the model will learn the representations of image.

##### Randomly Wired NN

In the paper [Randomly Wired Neural Network](https://arxiv.org/abs/1904.01569), it is proposed that NAS is meant for the architecture to be built freely with lack of human bias. However, existing NAS methods have limited the search methods and this paper attempts to let the architecture be build more freely using graph algorithms.

##### Datasets (ordered by magnitude)

- [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [ImageNET](https://www.image-net.org)

As the datasets are ordered by magnitude, they will be trained sequentially and if the classification accuracy reaches a threshhold then I will move on the next largest dataset.

### Objective

The objective of this project is for me to gain better understanding of the NAS in general and in the state-of-the-art techniques of NAS. As the project progresss, I plan to adapt the Randomly Wired Neural Network model to object detection as well.

### Tasks

- [x] Base NN architecture from which the Network Generator will build the model
- [x] Graph algorithm for the Network Generator
  - [x] Erdo ̋s-Re ́nyi (ER)
  - [x] Baraba ́si-Albert (BA)
  - [x] Watts-Strogatz (WS)
- [x] Network Generator (On-Going)
- [x] Dataset pipelines
- [x] Training Functions
- [x] Testing Functions
- [x] Script to train the model
- [x] Train on VOC 2012 dataset
- [ ] Train on ImageNet dataset

### Requirements

- You can install the required modules via commandline
  ```
  pip install -r requirements.txt
  ```

### Author
Seungwan Yoo / [@LinkedIn](https://www.linkedin.com/in/wanyoo2/)

### License
Under Apache License 2.0