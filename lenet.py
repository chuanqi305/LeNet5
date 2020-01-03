import os
import numpy as np

class Trainable(object):
    learning_rate = 0.0003
    learning_rate_decay = 0.001
    momentum = 0.95
    max_step = 500
    batch_size = 1000
    weight_decay = 0.001

    def __init__(self):
        self.weight_diff = 0
        self.bias_diff = 0

    def sgd_momentum(self, weight_diff, bias_diff):
        self.weight_diff = self.momentum * self.weight_diff + (1 - self.momentum) * weight_diff
        self.bias_diff = self.momentum * self.bias_diff + (1 - self.momentum) * bias_diff
        return self.weight_diff, self.bias_diff

class Conv(Trainable):
    def __init__(self, name, kernel, inc, outc):
        super(Conv, self).__init__()
        self.name = name
        self.kernel = kernel
        self.inc = inc
        self.outc = outc
        self.weight = np.random.randn(kernel, kernel, inc, outc) * np.sqrt(2.0 / (kernel * kernel * inc)) #msra
        self.bias = np.zeros(outc)

    def forward(self, x):
        self.x = x
        k = self.kernel
        n, h, w, c = x.shape
        h_out = h - (k - 1)
        w_out = w - (k - 1)
        weight = self.weight.reshape(-1, self.outc)
        output = np.zeros((n, h_out, w_out, self.outc))
        for i in range(h_out):
            for j in range(w_out):
                inp = x[:, i:i+k, j:j+k, :].reshape(n, -1)
                out = inp.dot(weight) + self.bias
                output[:, i, j, :] = out.reshape(n, -1)
        return output

    def backward(self, diff):
        n, h, w, c = diff.shape
        k = self.kernel
        h_in = h + (k - 1)
        w_in = w + (k - 1)

        weight_diff = np.zeros((k, k, self.inc, self.outc))
        for i in range(k):
            for j in range(k):
                #inp = (n, 28, 28, c) => (n*28*28, c) => (c, n*28*28)
                inp = self.x[:, i:i+h, j:j+w, :].reshape(-1, self.inc).T
                #diff = n, 28, 28, 6 => (n*28*28, 6)
                diff_out = diff.reshape(-1, self.outc)
                weight_diff[i, j, :, :] = inp.dot(diff_out)
        bias_diff = np.sum(diff, axis=(0, 1, 2))

        pad = k - 1
        diff_pad = np.pad(diff, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        rotated_weight = self.weight[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, self.inc)
        back_diff = np.zeros((n, h_in, w_in, self.inc))
        for i in range(h_in):
            for j in range(w_in):
                diff_out = diff_pad[:, i:i+k, j:j+k, :].reshape(n, -1)
                out = diff_out.dot(rotated_weight)
                back_diff[:, i, j, :] = out.reshape(n, -1)

        weight_diff, bias_diff = self.sgd_momentum(weight_diff, bias_diff)
        self.weight -= Trainable.learning_rate * weight_diff + Trainable.weight_decay * self.weight
        self.bias -= Trainable.learning_rate * bias_diff + Trainable.weight_decay * self.bias

        return back_diff
    
class Pooling():
    def forward(self, x):
        n, h, w, c = x.shape
        x_grid = x.reshape(n, h // 2, 2, w // 2, 2, c)
        out = np.max(x_grid, axis=(2, 4))
        self.mask = (out.reshape(n, h // 2, 1, w // 2, 1, c) == x_grid)
        return out

    def backward(self, diff):
        n, h, w, c = diff.shape
        diff_grid = diff.reshape(n, h, 1, w, 1, c)
        return (diff_grid * self.mask).reshape(n, h * 2, w * 2, c)

class ReLU():
    def forward(self, x):
        self.x = x
        return (x > 0) * x

    def backward(self, diff):
        return (self.x > 0) * diff

class FC(Trainable):
    def __init__(self, name, inc, outc):
        super(FC, self).__init__()
        self.name = name
        self.weight = np.random.randn(inc, outc) * np.sqrt(2.0 / inc) #msra
        self.bias = np.zeros(outc)

    def forward(self, x):
        self.origin_shape = x.shape
        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1)
        self.x = x
        return x.dot(self.weight) + self.bias
   
    def backward(self, diff):
        #diff = (n, 10)
        #self.x = (n, 1024) => (1024, n)
        weight_diff = self.x.T.dot(diff)
        bias_diff = np.sum(diff, axis=0)
        #weight = (1024, 10) => (10, 1024), back_diff = (n, 1024)
        back_diff = diff.dot(self.weight.T).reshape(self.origin_shape)

        weight_diff, bias_diff = self.sgd_momentum(weight_diff, bias_diff)
        self.weight -= Trainable.learning_rate * weight_diff + Trainable.weight_decay * self.weight
        self.bias -= Trainable.learning_rate * bias_diff + Trainable.weight_decay * self.bias
        return back_diff

class SoftmaxLoss():
    def forward(self, x):
        softmax = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
        self.softmax = softmax
        output = np.argmax(softmax, axis=1)
        if not hasattr(self, 'y'):
            return output

        y = self.y
        label = np.argmax(y, axis=1)
        loss = -np.sum(y * np.log(softmax) + (1 - y) * np.log(1 - softmax)) / len(y)
        accuracy = np.sum(output==label) / float(len(label))
        return loss, accuracy

    def backward(self, diff):
        return self.softmax - self.y

    def set_label(self, label):
        self.y = label

class LeNet:
    def __init__(self):
        conv1 = Conv("conv1", 5, 1, 6)
        pool1 = Pooling()
        relu1 = ReLU()
        conv2 = Conv("conv2", 5, 6, 16)
        pool2 = Pooling()
        relu2 = ReLU()
        fc3 = FC("fc3", 400, 120)
        relu3 = ReLU()
        fc4 = FC("fc4", 120, 84)
        relu4 = ReLU()
        fc5 = FC("fc5", 84, 10)
        loss = SoftmaxLoss()
        self.layers = [conv1, pool1, relu1, conv2, pool2, relu2, fc3, relu3, fc4, relu4, fc5, loss]

    def train(self, images, labels):
        index = 0
        batch_size = Trainable.batch_size
        for i in range(Trainable.max_step):
            x = images[index:index + batch_size] #mini batch sgd
            y = labels[index:index + batch_size]
            index += batch_size
            index = index % len(images)

            loss = self.layers[-1]
            loss.set_label(y)

            for layer in self.layers:
                x = layer.forward(x)
            print("step %d: loss=%.6f, accuracy=%.4f, lr=%g" % (i, x[0], x[1], Trainable.learning_rate))

            diff = 1.0
            for layer in reversed(self.layers):
                diff = layer.backward(diff)
            Trainable.learning_rate *= (1 - Trainable.learning_rate_decay)

    def predict(self, images):
        x = images
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def save(self, path):
        model = {}
        for layer in self.layers:
            if isinstance(layer, Trainable):
                model[layer.name] = {"w": layer.weight, "b": layer.bias}
        np.save(path, model)

    def load(self, path):
        model = np.load(path, allow_pickle=True).item()
        for layer in self.layers:
            if isinstance(layer, Trainable):
                layer.weight = model[layer.name]["w"]
                layer.bias = model[layer.name]["b"]
