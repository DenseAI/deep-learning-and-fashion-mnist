from keras import backend as K
from keras.layers import Layer
from keras import regularizers

import tensorflow as tf
import math

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, w_init=None, easy_margin=False, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
        self.w_init = w_init
        self.easy_margin = easy_margin

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        embedding, labels = inputs

        m = self.m
        s = self.s

        x = tf.nn.l2_normalize(embedding, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        cos_m = math.cos(m)
        sin_m = math.sin(m)

        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        # dot product
        cosine = x @ W

        sine = tf.sqrt(1.0 - tf.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > th, phi, cosine - mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s

        output = tf.nn.softmax(output)

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
