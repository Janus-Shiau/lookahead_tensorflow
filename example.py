'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
A very simple example code for using lookahead optimizer.
'''

import numpy as np
import tensorflow as tf

from lookahead_opt import BaseLookAhead

DATA_NUM = 100
DEPTH    = 10


if __name__ == "__main__":
    ### Generate Fake Inputs ###
    np_inputs = np.random.uniform(0, 1, (DATA_NUM, DEPTH))
    np_labels = np.random.uniform(0, 1, (DATA_NUM, 1))

    ### Build-up Network ###
    inputs  = tf.placeholder(dtype=tf.float32, shape=(1, DEPTH))
    labels  = tf.placeholder(dtype=tf.float32, shape=(1, 1))
    
    outputs = tf.layers.dense(inputs, 1)

    loss = tf.reduce_mean(tf.abs(outputs - labels))
    
    opt = tf.train.AdamOptimizer().minimize(loss)

    train_op = [opt, loss, outputs]

    with tf.Session() as sess:
        model_vars = [v for v in tf.trainable_variables()]
        tf.global_variables_initializer().run()

        lookahead = BaseLookAhead(model_vars, k=5, alpha=0.5)
        train_op += lookahead.get_ops()

        for step_idx in range(DATA_NUM):
            results = sess.run(train_op, feed_dict={
                inputs: np_inputs[[step_idx],:],
                labels: np_labels[[step_idx],:]
                })

            print ('L1 loss of step {}: {}'.format(step_idx, results[1]))
  

