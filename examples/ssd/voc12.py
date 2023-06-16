from bs4 import BeautifulSoup

import cv2
import numpy as np
import os
import tensorflow as tf

VOC12_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                 'sheep', 'sofa', 'train', 'tvmonitor']

class VOC12Dataset():
    def __init__(self, root_folder, filename, image_size=300) -> None:
        self.root_folder = root_folder
        self.filename = filename
        self.image_size = image_size

        file_path = os.path.join(self.root_folder, 'ImageSets', 'Main', self.filename)
        with open(file_path) as f:
            sample_names = f.readlines()

        self.sample_names = [f.rstrip() for f in sample_names]

    def _get_annotation(self, filename):
        xml_fpath = os.path.join(self.root_folder, 'Annotations', filename + '.xml')

        if not os.path.exists(xml_fpath):
            raise FileNotFoundError(f"Annotation file: {filename} does not exist")

        with open(xml_fpath, 'r') as xf:
            xml_file = xf.read() 

        bs_xml_file = BeautifulSoup(xml_file, 'xml')

        size = bs_xml_file.find('size')
        width = int(size.find('width').get_text())
        height = int(size.find('height').get_text())

        boxes = []
        for objs in bs_xml_file.find_all('object'):
            label = objs.find('name').get_text()
            label_idx = VOC12_CLASSES.index(label)

            bndbox = objs.find('bndbox')
            xmin = int(bndbox.find('xmin').get_text()) / width
            xmax = int(bndbox.find('xmax').get_text()) / width
            ymin = int(bndbox.find('ymin').get_text()) / height
            ymax = int(bndbox.find('ymax').get_text()) / height

            boxes.append([xmin, ymin, xmax, ymax, label_idx])

        return boxes

    def _get_sample(self, x):
        """
        Images are resized to (image_size, image_size) with
        range 0-1. Boxes are also normalized to 0-1.
        """
        filename = bytes.decode(x.numpy())
        
        image_path = os.path.join(self.root_folder, 'JPEGImages', filename + '.jpg')
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        boxes = self._get_annotation(filename)
        return tf.image.convert_image_dtype(image, tf.float32), tf.ragged.constant(boxes)

    def get(self, batch_size=8) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(self.sample_names)
        dataset = dataset.map(lambda x: tf.py_function(self._get_sample, [x], [tf.float32, tf.RaggedTensorSpec(shape=[None, 5], dtype=tf.float32)]))
        dataset = dataset.batch(batch_size)
        return dataset