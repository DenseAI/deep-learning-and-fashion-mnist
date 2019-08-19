from keras import backend as K
from keras.layers import Layer
from keras import regularizers

import tensorflow as tf
import math

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=64.0, m=0.50, regularizer=None, w_init=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
        self.w_init = w_init

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        embedding, labels = inputs
        # c = K.shape(x)[-1]
        # # normalize feature
        # x = tf.nn.l2_normalize(x, axis=1)
        # # normalize weights
        # W = tf.nn.l2_normalize(self.W, axis=0)
        # # dot product
        # logits = x @ W
        # # add margin
        # # clip logits to prevent zero division when backward
        # theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        # target_logits = tf.cos(theta + self.m)
        # # sin = tf.sqrt(1 - logits**2)
        # # cos_m = tf.cos(logits)
        # # sin_m = tf.sin(logits)
        # # target_logits = logits * cos_m - sin * sin_m
        # #
        # logits = logits * (1 - y) + target_logits * y
        # # feature re-scale
        # logits *= self.s
        # out = tf.nn.softmax(logits)

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m  # issue 1
        threshold = math.cos(math.pi - self.m)
        #with tf.variable_scope('arcface_loss'):

        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], self.n_classes),
                                  initializer=self.w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(K.cast(labels, dtype='int32'), depth=self.n_classes, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)



class SphereFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)



class SoftMax(Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SoftMax, self).__init__(**kwargs)
        self.n_classes = n_classes
        #self.s = s
        #self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(SoftMax, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        logits = tf.matmul(x, W)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        return loss

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
