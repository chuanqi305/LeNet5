# LeNet5
Pure numpy implementation of LeNet5, to help you understand how CNN works.
### Prerequisite
- python3
- numpy
- gzip
- urllib
```
pip install numpy
pip install gzip
pip install urllib
```
### Notes
The MNIST dataset will automatically be downloaded from http://yann.lecun.com/exdb/mnist/ as needed.

### Run
1. Run a training:
```
python train.py
```
2. Evaluate the accuracy on the validation dataset:
```
python eval.py
```
