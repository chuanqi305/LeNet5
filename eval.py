import numpy as np
from lenet import LeNet
import mnist_loader

net = LeNet()
net.load("lenet_weights.npy")

images, labels = mnist_loader.load_valid()

predict = net.predict(images)
tp = np.sum(predict == labels)
accuracy = float(tp) / len(labels)
print("accuracy=%f" % accuracy)
