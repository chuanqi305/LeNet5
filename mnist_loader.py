# reference: http://yann.lecun.com/exdb/mnist/
import numpy as np
import matplotlib.pyplot as plt

import os.path
import gzip
import urllib.request

def load_train(input_path = 'mnist_raw'):
    images_file = 'train-images-idx3-ubyte.gz'
    labels_file = 'train-labels-idx1-ubyte.gz'
    return load(input_path, images_file, labels_file)

def load_valid(input_path = 'mnist_raw'):
    images_file = 't10k-images-idx3-ubyte.gz'
    labels_file = 't10k-labels-idx1-ubyte.gz'
    return load(input_path, images_file, labels_file)

def download(download_dir, filename):
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    url = "http://yann.lecun.com/exdb/mnist/" + filename
    targetfile = os.path.join(download_dir, filename)
    print("Downloading: " + filename + " ...")
    urllib.request.urlretrieve(url, targetfile)

def load(input_path, images_file, labels_file):
    images_path = os.path.join(input_path, images_file)
    labels_path = os.path.join(input_path, labels_file)
    if not os.path.isfile(images_path):
        download(input_path, images_file)
    if not os.path.isfile(labels_path):
        download(input_path, labels_file)

    images_byte = gzip.open(images_path, 'r')
    labels_byte = gzip.open(labels_path, 'r')

    # read images
    magic_num  = int.from_bytes(images_byte.read(4),  byteorder='big')
    num_images = int.from_bytes(images_byte.read(4),  byteorder='big')
    img_height = int.from_bytes(images_byte.read(4),  byteorder='big')
    img_width  = int.from_bytes(images_byte.read(4),  byteorder='big')
    assert(magic_num == 2051)
    assert(img_height == 28)
    assert(img_width == 28)

    images = np.zeros((num_images, 32, 32, 1), dtype=np.float32)
    for i in range(num_images):
        data_raw = images_byte.read(img_height * img_width)
        data_np  = np.frombuffer(data_raw, dtype=np.uint8)
        images[i,2:-2,2:-2,0] = data_np.reshape(28,28) / 255.

    # read labels
    magic_num  = int.from_bytes(labels_byte.read(4),  byteorder='big')
    num_labels = int.from_bytes(labels_byte.read(4),  byteorder='big')
    assert(magic_num == 2049)
    assert(num_labels == num_images)

    buf = labels_byte.read(num_labels)    
    labels = np.frombuffer(buf, dtype=np.uint8)

    return images, labels

def show_images(images, labels):
    cols = 5
    rows = 2

    plt.figure(figsize=(10,7))
     
    for i in range(cols*rows):
        plt.subplot(rows, cols, i+1)        
        plt.imshow(images[i,:,:,0], cmap="gray")
        plt.title("label: " + str(labels[i]))
        plt.axis('off')

if __name__ == '__main__':
    images, labels = load_train()
    show_images(images, labels)
