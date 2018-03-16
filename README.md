# chainer-cnn
Simple Convolutional Neural Network for sentence classification (positive or negative) .


# Requirements
This code is written in Python with Chainer which is framework of Deep Neural Network.  
Please download `latest-ja-word2vec-gensim-model` from [this site](http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip) and put it in the 'shiroyagi' directory as these codes.  
ex) shiroyagi/latest-ja-word2vec-gensim-model/word2vec.gensim.model
					     word2vec.gensim.model.syn1neg.npy
					     word2vec.gensim.model.wv.syn0.npy

Please Modify L.Convolution2D Parameter and F.max_pooling_2d Parameter in SimpleCNN.py  
as word2vec list data which is resulted from your input documents matches.


# Usage
```
  $ python train_cnn.py [--gpu 1 or 0]   
```

# Optional arguments
```
  -h, --help            show this help message and exit
  --gpu   GPU           1: use gpu, 0: use cpu
  --data  DATA          an input data file
  --epoch EPOCH         number of epochs to learn
  --batchsize BATCHSIZE
                        learning minibatch size
  --nunits NUNITS       number of units
```

# Data format for input data
  - [0 or 1] [Sequence of words]  
    - 1 and 0 are positive and negative, respectively.  

## Examples
```
1 That was so beautiful that it can't be put into words . (POSITIVE SETENCE)
0 I do not want to go to school because I do like to study math . (NEGATIVE SENTENCE)
```
