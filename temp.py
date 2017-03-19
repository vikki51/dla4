import tensorflow as tf
import numpy as np

w = tf.Variable(tf.random_normal([784, 256]))

saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver.restore(sess,'D:\Downloads\DL\Assignment4\\tempModel.ckpt') 	 	

w1 = sess.run(w)

print(w1[0][0])