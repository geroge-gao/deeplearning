# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
input_vec_size = lstm_size = 28 # 输入向量的维度
time_step_size = 28 # 循环层长度

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    # XT shape: (time_step_size, batch_size, input_vec_size)
    #对这一步操作还不是太理解，为什么需要将第一行和第二行置换
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size,[28, 128, 28]


    # XR shape: (time_step_size * batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)

    # Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]


    # Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    # rnn..static_rnn()的输出对应于每一个timestep，如果只关心最后一步的输出，取outputs[-1]即可
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)  # 时间序列上每个Cell的输出:[... shape=(128, 28)..]

    # tanh activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the state

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 读取数据

# mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵
#训练集包含55000张图片，每张图片为28*28维矩阵
#训练集标签同样对应55000,10表示存在10个标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 将每张图用一个28x28的矩阵表示,(55000,28,28,1)
#-1表示该数未知，根据后面28*28,将trx分成55000个28*28的矩阵，每个表示一张图片。

trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

# get lstm_size and output 10 labels
#生成一个初始随机值
W = init_weights([lstm_size, 10])  # 输出层权重矩阵28×10
B = init_weights([10])  # 输出层bais

#
py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#返回每一行的最大值
predict_op = tf.argmax(py_x, 1)

#tf.ConfigProto，一般是在创建session时对session进行配置
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True#允许gpu在使用的过程中慢慢增加。

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(100):
        #从训练集中每段选择一个batch训练，batch_size= end-start
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            #X (128,28,28)
        s=len(teX)
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices]})))