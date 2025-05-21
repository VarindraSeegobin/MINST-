from cProfile import label

import numpy as np
import os
import struct
from array import array


class DataLoader:
    def __init__(self, training_label_path, training_data_path, testing_label_path, testing_data_path):
        self.training_label_path = training_label_path
        self.training_data_path = training_data_path
        self.testing_label_path = testing_label_path
        self.testing_data_path = testing_data_path

    def load_labels_and_data(self, data_path, label_path):
        labels = []
        images = []
        #Open the training data labels
        with open(label_path, mode='rb') as file:
            magic, size = struct.unpack('>II', file.read(8))
            if magic != 2049:
                raise ValueError('Invalid magic number. Got' + str(magic))
            labels = array('B', file.read())

        #Open the training data
        with open(data_path, mode='rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        return images, labels


    def load_training(self):
        imgs, labels = self.load_labels_and_data(self.training_data_path, self.training_label_path)
        return imgs, labels

    def load_testing(self):
        imgs, labels = self.load_labels_and_data( self.testing_data_path, self.testing_label_path)
        return imgs, labels
