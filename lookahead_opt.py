'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The implementation of lookahead optimization strategy.

    https://arxiv.org/abs/1907.08610

"Lookahead Optimizer: k steps forward, 1 step back"
Mockael R. Zhang, Jamses Lucas, Geoffrey Hinton, Jimmy Ba.
arXiv preprint.
'''
import tensorflow as tf


class BaseLookAhead():
    """ Lookahead optimization strategy for any optimizer.
        This implemention is based on:
        
            https://arxiv.org/abs/1907.08610

        "Lookahead Optimizer: k steps forward, 1 step back"
        Mockael R. Zhang, Jamses Lucas, Geoffrey Hinton, Jimmy Ba
    """
    def __init__(self, model_vars, k=5, alpha=0.5, name='lookahead'):
        """ [Args]
                k: the difined forward step k. [int]
                alpha: the defined learning rate for lookahead. [float]
                name: namescope. [str]
        """
        self.is_injected  = False
        self.name         = name
        self.variables    = []
        self.update_ops   = []

        self._inject(model_vars, k, alpha)
            

    def get_ops(self):
        """ Returns the update operators for the weights. 

            Please run this operators in session or add this to training operation. 
            [Returns]
                list (slow/fast weight) of lists (for each weights) of operations.
        """
        if not self.is_injected:
            raise AttributeError("LookAhead have not been injected!!")

        return [self.slow_weights_op, self.fast_weights_op]


    def _inject(self, model_vars, k, alpha):
        """ Inject the required update ops with all trainable variables.
            [Args]
                model_vars: list of trainable tf.Variables. [list]
        """
        with tf.variable_scope(self.name) as scope:
            counter      = tf.Variable(0, trainable=False, dtype=tf.int32, name='counter')

            self.k       = tf.constant(k, dtype=tf.int32, name='reset_step_k')
            self.alpha   = tf.constant(alpha, dtype=tf.float32, name='alpha')
            self.counter = tf.cond(
                tf.equal(counter, self.k), 
                lambda: tf.assign(counter, 0), 
                lambda: tf.assign(counter, counter+1))

            with tf.variable_scope('weights') as scope:
                for weight in model_vars:
                    slow_weight   = tf.Variable(weight.eval(), trainable=False, dtype=tf.float32, name='slow_weight')
                    update_weight = tf.identity(slow_weight + (weight - slow_weight) * self.alpha, name='update_weight')

                    self.variables.append(slow_weight)
                    self.update_ops.append(update_weight)

            self.slow_weights_op = tf.cond(
                tf.equal(self.counter, self.k), 
                lambda: self._assign_list(self.variables, self.update_ops), 
                lambda: self._assign_list(self.variables, self.variables),
                name='cond_update_slow_weight')

            self.fast_weights_op = tf.cond(
                tf.equal(self.counter, self.k), 
                lambda: self._assign_list(model_vars, self.slow_weights_op), 
                lambda: self._assign_list(model_vars, model_vars),
                name='cond_update_fast_weight')

        
        tf.initialize_variables(self.variables + [counter]).run()

        self.is_injected = True
    
    
    def _assign_list(self, list_1, list2):
        """ Tensorflow assign function for list of tensors. """
        assign_results = []
        for (list_1,list2) in zip(list_1, list2):
            assign_results.append(tf.assign(list_1, list2))
        
        return assign_results
    