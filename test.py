#_*_coding:utf8_*_
import cv2
import numpy as np
import os
import random
from lenet import LeNet

net = LeNet()
net.load("lenet.npy")
test_dir = "mnist/test"
files = os.listdir(test_dir)

random.shuffle(files)
print("press ESC to exit")
for f in files:
    img = cv2.imread(os.path.join(test_dir, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    x = img.astype(np.float32).reshape(1, 32, 32, 1) / 255.0
    predict = net.predict(x)
    title = str(int(predict[0]))

    img = cv2.resize(img, (256, 256))
    cv2.putText(img, title, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.imshow("lenet", img)
    k = cv2.waitKey(0)
    if k == 27:
        break

