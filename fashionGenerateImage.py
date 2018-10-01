import cv2 as cv

import os
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


if __name__ == '__main__':
    # GENERATION IMAGE D'ENTRAINEMENT | default path ./
    x, y = load_mnist('.')
    for index, image in enumerate(x):
        if not os.path.exists(str(y[index])):
            os.makedirs(str(y[index]))
        filename = str(y[index]) + "/image" + str(index) + ".png"
        cv.imwrite(filename, image.reshape(28, 28))
    # GENERATION IMAGE DE TEST | default path ./test
    xTest, yTest = load_mnist('.', 't10k')
    for index, image in enumerate(xTest):
        # écriture du fichier
        if not os.path.exists("test/" + str(yTest[index])):
            os.makedirs("test/" + str(yTest[index]))
        filename = "test/" + str(yTest[index]) + "/image" + str(index) + ".png"
        cv.imwrite(filename, image.reshape(28, 28))
