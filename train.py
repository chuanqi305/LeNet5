from lenet import LeNet
import mnist_loader

if __name__ == '__main__':
    images, labels = mnist_loader.load_train()
    lenet = LeNet()
    lenet.train(images, labels)
    lenet.save("lenet_weights.npy")
