# lookahead_tensorflow
Lookahead optimizer (["Lookahead Optimizer: k steps forward, 1 step back"](https://arxiv.org/abs/1907.08610)) for tensorflow

### Environment 
This code is implemmented and tested with [tensorflow](https://www.tensorflow.org/) 1.11.0. \
I didn't use any spetial operator, so it should also work for other version of tensorflow.

### Usage
1. Please assert the class after all variable initialization, and initialize the BaseLoookAhead with all trainable variables.
```
import tensorflow as tf
from lookahead_opt import BaseLookAhead

"""
Build your model here
"""

model_vars = [v for v in tf.trainable_variables()]
tf.global_variables_initializer().run()

lookahead = BaseLookAhead(model_vars)
```

2. Add the assign operator to training operation or directly run in session.

```
# Add to train_op
train_op += lookahead.get_ops()

# Or just run the Session
with tf.Session() as sess:
  _ = sess.run(lookahead.get_ops())
```

### Details
TBA.

### Contact & Copy Right
Code work by Jia-Yau Shiau <jiayau.shiau@gmail.com>.
