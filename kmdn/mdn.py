from keras import backend as K
from keras.layers import Dense, Input, merge
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import math

class MDN(Layer):
    def __init__(self, output_dim, num_mix, kernel='unigaussian', **kwargs):
        self.output_dim = output_dim
        self.kernel = kernel
        self.num_mix = num_mix
        
        with tf.name_scope('MDNLayer'):
            # self.inputs      = Input(shape=(input_dim,), dtype='float32', name='msn_input')
            self.mdn_mus     = Dense(self.num_mix * self.output_dim, name='mdn_mus')#(self.inputs)
            self.mdn_sigmas  = Dense(self.num_mix, activation=K.exp, name='mdn_sigmas')#(self.inputs)
            self.mdn_pi      = Dense(self.num_mix, activation=K.softmax, name='mdn_pi')#(self.inputs)
            # self.mdn_out     = merge([self.mdn_mus, self.mdn_sigmas, self.mdn_pi], mode='concat', name='mdn_out')

        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.input_shape = input_shape

        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        
        self.trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        

        self.non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        # self.updates = self.mdn_mus.updates + self.mdn_sigmas.updates + self.mdn_pi.updates
        # self.regularizers = self.mdn_mus.regularizers + self.mdn_sigmas.regularizers + self.mdn_pi.regularizers
        # self.constraints = self.mdn_mus.constraints + self.mdn_sigmas.constraints + self.mdn_pi.constraints


        self.built = True

    def call(self, x, mask=None):
        m = self.mdn_mus(x)
        s = self.mdn_sigmas(x)
        p = self.mdn_pi(x)

        with tf.name_scope('MDNLayer'):
            mdn_out = merge([m, s, p], mode='concat', name='mdn_out')
        return mdn_out
        

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = {'output_dim': self.output_dim,                                    
                #   'input_shape': self.input_shape,
                  'num_mix': self.num_mix,
                  'kernel': self.kernel}
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_loss_func(self):
        def unigaussian_loss(y_true, y_pred):
            mix = tf.range(start = 0, limit = self.num_mix)
            out_mu, out_sigma, out_pi = tf.split_v(split_dim=1, size_splits=[self.num_mix * self.output_dim, self.num_mix, self.num_mix], value=y_pred, name='mdn_coef_split')
            # tf.to_float(out_mu)
            # print('----- ', tf.shape(y_pred)[0].eval(session=K.get_session()))
            # print('----- ', tf.shape(y_pred)[1])/
            
            def loss_i(i):         
                batch_size = tf.shape(out_sigma)[0]
                sigma_i = tf.slice(out_sigma, [0, i], [batch_size, 1], name='mdn_sigma_slice')
                pi_i = tf.slice(out_pi, [0, i], [batch_size, 1], name='mdn_pi_slice')        
                mu_i = tf.slice(out_mu, [0, i * self.output_dim], [batch_size, self.output_dim], name='mdn_mu_slice')

                print('***.....>> ', i * self.output_dim)
                tf.Print(mu_i, [i], ">>>>>>>  ")
                # print('.....>> ', tf.shape(y_true))

                dist = tf.contrib.distributions.Normal(mu=mu_i, sigma=sigma_i)
                loss = dist.pdf(y_true)

                # loss = gaussian_kernel_(y_true, mu_i, sigma_i)

                loss = pi_i * loss

                return loss

            result = tf.map_fn(lambda  m: loss_i(m), mix, dtype=tf.float32, name='mix_map_fn')
            
            result = tf.reduce_sum(result, axis=0, keep_dims=False)
            result = -tf.log(result)
            # result = tf.reduce_mean(result, axis=1)
            result = tf.reduce_mean(result)
            # result = tf.reduce_sum(result)
            return result

        if self.kernel == 'unigaussian':
            with tf.name_scope('MDNLayer'):
                return unigaussian_loss
    
oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def gaussian_kernel_(y, mu_i, sigma_i):
    print('----->> ', y.get_shape())
    print('----->> ', mu_i.get_shape())
    result = tf.subtract(y, mu_i)
    result = tf.multiply(result,tf.reciprocal(sigma_i))
    result = -tf.square(result)/2
    return tf.mul(tf.exp(result),tf.reciprocal(sigma_i))*oneDivSqrtTwoPI
