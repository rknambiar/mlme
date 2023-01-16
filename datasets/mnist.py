import numpy as np
import os
import struct

class MNIST_DATASET:
    """ MNIST dataset loader. Unzip .gz file before reading """
    
    @classmethod
    def load_dataset(cls, dataset_path: str, 
                    reshape: bool = False):
        """ Read MNIST data from local folder """
        train_images_path = os.path.join(dataset_path, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(dataset_path, 'train-labels-idx1-ubyte')
        test_images_path = os.path.join(dataset_path, 't10k-images-idx3-ubyte')
        test_labels_path = os.path.join(dataset_path, 't10k-labels-idx1-ubyte')

        x_train, y_train = cls.read_data(train_images_path, train_labels_path)
        x_test, y_test = cls.read_data(test_images_path, test_labels_path)

        if reshape:
            x_train = x_train.reshape(-1, 28, 28)
            x_test = x_test.reshape(-1, 28, 28)

        return x_train, y_train, x_test, y_test

    @classmethod
    def read_data(cls, image_path: str, label_path: str):
        """ Read images and labels file from path """
        with open(label_path, "rb") as lbpath:
            magic, n = struct.unpack(">II", lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        
        with open(image_path, "rb") as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        
        return images, labels
