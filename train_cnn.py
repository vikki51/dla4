'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
                        strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_1 =     tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x_1, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.placeholder(tf.int32, [None, 10])
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
saver= tf.train.Saver()
sess = tf.InteractiveSession()

def train(trainX, trainY):
    '''
    Complete this function.
    '''

    tf.global_variables_initializer().run()
    # Train
    batch_size=100
    for j in range(10):
        epochloss=0
        for i in range(int(trainX.shape[0]/batch_size)):
           
            n = min(batch_size, trainX.shape[0]-i*batch_size)


            batch_xs= trainX[i*batch_size:(i)*batch_size+n]

            batch_xs=batch_xs.reshape((n,784))
            batch_ys= trainY[i*batch_size:(i)*batch_size+n]
            batch_ys=np.eye(10)[batch_ys]
            # print(batch_ys)
            sess.run(train_step, feed_dict={x_1: batch_xs, y_: batch_ys,keep_prob:0.5})
            loss=sess.run(cross_entropy,feed_dict={x_1: batch_xs, y_: batch_ys,keep_prob:0.5})
            epochloss+=loss
            print (i,loss)
    saver.save(sess, 'D:\Downloads\DL\Assignment4\\secondcnn.ckpt')
def test(testX):
    saver = tf.train.import_meta_graph('D:\Downloads\DL\Assignment4\\secondcnn.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    testX=testX.reshape((testX.shape[0],784))
    output=sess.run(y_conv, feed_dict={x_1: testX,keep_prob:1.0})
    output=np.argmax(output,1)
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    return output

