# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class SimpleCNN(Chain):
    
    def __init__(self, input_channel, output_channel, filter_height, filter_width, mid_units, n_units, n_label):
        super(SimpleCNN, self).__init__(
           conv1 = L.Convolution2D(input_channel, 16, (4, 4), 3, 1),
           conv2 = L.Convolution2D(16, 16, (1, 1), 1, 0),
           conv3 = L.Convolution2D(16, 32, (3, 3), 1, 1),
           conv4 = L.Convolution2D(32, 32, (3, 3), 1, 1),
           conv5 = L.Convolution2D(32, 64, (3, 3), 1, 1),
           conv6 = L.Convolution2D(64, output_channel, (3, 3), 1, 1),
            l1    = L.Linear(mid_units, n_units),
            l2    = L.Linear(n_units,  n_label)
        )
    
    #Classifier によって呼ばれる
    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), (4,3), 3, 2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.max_pooling_2d(F.relu(self.conv4(h3)), (3, 3), 3, 1)
        h5 = F.relu(self.conv5(h4))
        h6 = F.max_pooling_2d(F.relu(self.conv6(h5)), (3, 5), 3, 1)
        h7 = F.dropout(F.relu(self.l1(h6)))
        y = F.dropout(self.l2(h7))
        return y

    def forward(self, x, t, train=True):
        h1 = F.relu(self.conv1(x))
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), (4,3), 3, 2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.max_pooling_2d(F.relu(self.conv4(h3)), (3, 3), 3, 1)
        h5 = F.relu(self.conv5(h4))
        h6 = F.max_pooling_2d(F.relu(self.conv6(h5)), (3, 5), 3, 1)
        h7 = F.dropout(F.relu(self.l1(h6)))
        y = F.dropout(self.l2(h7))
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
