import xml.etree.ElementTree as ET
import torch.utils.data as data
import os
import numpy as np
from PIL import Image

VOC_CLASSES = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class VocDataset(data.Dataset):
    def __init__(self, data_path, dataset_split, transform):
        self.data_path = data_path
        self.transform = transform
        self.dataset_split = dataset_split

        self.__init_classes()
        (
            self.names,
            self.labels,
            self.box_indices,
            self.label_order,
        ) = self.__dataset_info()

    def __getitem__(self, index):
        x = Image.open(self.data_path + "/JPEGImages/" +
                       self.names[index] + ".jpg")

        scale = np.random.rand() * 2 + 0.25
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)
        if min(w, h) < 224:
            scale = 224 / min(w, h)
            w = int(x.size[0] * scale)
            h = int(x.size[1] * scale)

        x = self.transform(x)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.names)

    def __init_classes(self):
        self.classes = VOC_CLASSES
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    def __dataset_info(self):
        with open(
            self.data_path + "/ImageSets/Main/" + self.dataset_split + ".txt"
        ) as f:
            annotations = f.readlines()

        annotations = [n[:-1] for n in annotations]
        box_indices = []
        names = []
        labels = []
        label_order = []
        for af in annotations:
            filename = os.path.join(self.data_path, "Annotations", af)
            tree = ET.parse(filename + ".xml")
            objs = tree.findall("object")
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.int32)
            boxes_cl = np.zeros((num_objs), dtype=np.int32)
            boxes_cla = []
            temp_label = []
            for ix, obj in enumerate(objs):
                bbox = obj.find("bndbox")
                # Make pixel indexes 0-based
                x1 = float(bbox.find("xmin").text) - 1
                y1 = float(bbox.find("ymin").text) - 1
                x2 = float(bbox.find("xmax").text) - 1
                y2 = float(bbox.find("ymax").text) - 1

                cls = self.class_to_ind[obj.find("name").text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                boxes_cl[ix] = cls
                boxes_cla.append(boxes[ix, :])
                temp_label.append(cls)
            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)
            box_indices.append(boxes_cla)
            label_order.append(temp_label)

        return (
            np.array(names),
            np.array(labels).astype(np.float32),
            np.array(box_indices, dtype=object),
            label_order,
        )
