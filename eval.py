import os
import cv2
import numpy as np
from lenet import LeNet

data_dir = "mnist/test"
net = LeNet()
net.load("lenet.npy")
files = os.listdir(data_dir)
images = []
labels = []
for f in files:
    img = cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32).reshape(32, 32, 1) / 255.0
    images.append(img)
    labels.append(int(f[0]))

x = np.array(images)
y = np.array(labels)

predict = net.predict(x)
tp = np.sum(predict == y)
accuracy = float(tp) / len(files)
print("accuracy=%f" % accuracy)
