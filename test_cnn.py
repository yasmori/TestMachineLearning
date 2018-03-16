# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils
from gensim.models import word2vec

import util
from SimpleCNN import SimpleCNN

"""
CNNによるテキスト分類 (posi-nega)
 - 5層のディープニューラルネット
 - 単語ベクトルにはWordEmbeddingモデルを使用
"""

#引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,            help='1: use gpu, 0: use cpu')
parser.add_argument('--data '    , dest='data'       , type=str, default='input.dat',  help='an input data file')
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=40,           help='learning minibatch size')
parser.add_argument('--nunits'   , dest='nunits'     , type=int, default=200,          help='number of units')

args = parser.parse_args()
batchsize   = args.batchsize    # minibatch size
n_epoch     = args.epoch        # エポック数(パラメータ更新回数)

# Prepare dataset
dataset, height, width = util.load_data(args.data)
print('height:', height)
print('width:', width)

dataset['source'] = dataset['source'].astype(np.float32) #特徴量
dataset['target'] = dataset['target'].astype(np.int32) #ラベル

x_test = dataset['source']
y_test = dataset['target']
N_test = y_test.size         # test data size

print('len(x_test):', len(x_test))
print('len(y_test):', len(y_test))

# (nsample, channel, height, width) の4次元テンソルに変換
input_channel = 1
x_test  = x_test.reshape(len(x_test), input_channel, height, width)

# 隠れ層のユニット数
n_units = args.nunits
n_label = 2
filter_height = 3
#output_channel = 50
output_channel = 128
mid_units = 256

#モデルの定義
model = L.Classifier( SimpleCNN(input_channel, output_channel, filter_height, width, mid_units, n_units, n_label))
# "pn_classifier_cnn.model"の情報をmodelに読み込む
#serializers.load_npz('./model/pn_classifier_cnn.model', model) 

#GPUを使うかどうか
xp = np
if args.gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

batchsize = args.batchsize
n_epoch = args.epoch

# Setup optimizer
optimizer = optimizers.AdaGrad()
optimizer.setup(model)

# " model,optimizerに読み込む
serializers.load_npz('./model/pn_classifier_cnn.model', model)
serializers.load_npz('./model/pn_classifier_cnn.state', optimizer)

# evaluation
sum_test_loss     = 0.0
sum_test_accuracy = 0.0

print('fN_test ', N_test )
print('batchsize ', batchsize)
for i in six.moves.range(0, N_test, batchsize):

    # all test data
    x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
    t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))

    loss = model(x, t)

    sum_test_loss     += float(loss.data) * len(t.data)
    sum_test_accuracy += float(model.accuracy.data)  * len(t.data)

print(' test mean loss={}, accuracy={}'.format(
    sum_test_loss / N_test, sum_test_accuracy / N_test)) #平均誤差

sys.stdout.flush()


