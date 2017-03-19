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
W = tf.Variable(tf.random_normal([784, 200]))
b = tf.Variable(tf.zeros([200]))
V = tf.Variable(tf.random_normal([200, 10]))
b2= tf.Variable(tf.zeros([10]))
def forward_test(x):
    hs= tf.matmul(x, W) + b
    ha= tf.nn.relu(hs)
    y= tf.matmul(ha, V) + b2
    return y
def train(trainX, trainY):
    '''
    Complete this function.
    '''


  # Create the model

    x = tf.placeholder(tf.float32, [None, 784])

    hs= tf.matmul(x, W) + b
    ha= tf.nn.relu(hs)
    y= tf.matmul(ha, V) + b2
    # Define loss and optimizer
    y_ = tf.placeholder(tf.int32, [None, 10])
    # saver = tf.train.Saver([W,b,V,b2])
    saver= tf.train.Saver()
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    batch_size=100
    for j in range(15):
        epochloss=0
        for i in range(int(trainX.shape[0]/batch_size)):
           
            n = min(batch_size, x.shape[0]-i*batch_size)


            batch_xs= trainX[i*batch_size:(i)*batch_size+n]

            batch_xs=batch_xs.reshape((n,784))
            batch_ys= trainY[i*batch_size:(i)*batch_size+n]
            batch_ys=np.eye(10)[batch_ys]
            # print(batch_ys)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            loss=sess.run(cross_entropy,feed_dict={x: batch_xs, y_: batch_ys})
            epochloss+=loss
        print (epochloss)
    saver.save(sess, 'D:\Downloads\DL\Assignment4\\my-model.ckpt')
   
    


def test(testX):
    sess = tf.InteractiveSession()
    #saver = tf.train.Saver()
    # W = tf.Variable(tf.random_normal([784, 200]))
    # b = tf.Variable(tf.zeros([200]))
    # V = tf.Variable(tf.random_normal([200, 10]))
    # b2= tf.Variable(tf.zeros([10]))
    saver = tf.train.import_meta_graph('D:\Downloads\DL\Assignment4\\my-model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    #saver.restore(sess, 'D:\Downloads\DL\Assignment4\\my-model.ckpt')
    x = tf.placeholder(tf.float32, [None, 784])
    hs= tf.matmul(x, W) + b
    ha= tf.nn.relu(hs)
    y= tf.matmul(ha, V) + b2
    # forward_test(x)
    testX=testX.reshape((testX.shape[0],784))
    # print (testX.shape)
    Y=sess.run(y,feed_dict={x:testX})
    print(Y.shape)
    output=np.zeros((Y.shape[0]))
    # print(output.shape)
    for i in range(Y.shape[0]):
        # print(Y[i])
        # print("------")
        output[i]=np.argmax(Y[i])
    
    print (output)
    return output
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    # return np.zeros(testX.shape[0])
