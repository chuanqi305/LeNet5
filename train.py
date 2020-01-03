import os
import random
import cv2
import numpy as np
from lenet import LeNet

def get_training_data(data_dir):
    images = []
    labels = []
    files = os.listdir(data_dir)
    random.shuffle(files)
    for f in files:
        img = cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = img.astype(np.float32).reshape(32, 32, 1) / 255.0
        images.append(img)
        num = int(f[0])
        label = np.zeros(10, dtype=np.float32)
        label[num] = 1
        labels.append(label)
    return (np.array(images), np.array(labels))

if __name__ == '__main__':
    x, y = get_training_data("mnist/train")
    lenet = LeNet()
    lenet.train(x, y)
    lenet.save("lenet.npy")
