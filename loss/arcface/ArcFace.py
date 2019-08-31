from keras import backend as K
from keras.layers import Layer
from keras import regularizers
from keras.engine import InputSpec

import tensorflow as tf
import math

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, w_init=None, easy_margin=True, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
        self.w_init = w_init
        self.easy_margin = easy_margin

    def build(self, input_shape):
        print("input_shape: ", input_shape)
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

        label_onehot = tf.one_hot(K.cast(labels, dtype='int32'), self.n_classes)
        output = (label_onehot * phi) + ((1.0 - label_onehot) * cosine)
        output *= self.s

        output = tf.nn.softmax(output)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)



class SphereFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, w_init=None, easy_margin=False, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)
        self.w_init = w_init
        self.easy_margin = easy_margin

    def build(self, input_shape):
        print("input_shape: ", input_shape)
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
        label_onehot = tf.one_hot(K.cast(y, dtype='int32'), self.n_classes)
        logits = logits * (1 - label_onehot) + target_logits * (label_onehot)

        #label_onehot = y #tf.one_hot(K.cast(y, dtype='int32'), self.n_classes)
        #logits = logits * (1.1 - label_onehot * 1) + target_logits * (label_onehot * 1 + 0.1)
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

# class SphereFace(Layer):
#     def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
#         super(SphereFace, self).__init__(**kwargs)
#         self.n_classes = n_classes
#         self.s = s
#         self.m = m
#         self.regularizer = regularizers.get(regularizer)
#
#     def build(self, input_shape):
#         super(SphereFace, self).build(input_shape[0])
#         self.W = self.add_weight(name='W',
#                                 shape=(input_shape[0][-1], self.n_classes),
#                                 initializer='glorot_uniform',
#                                 trainable=True,
#                                 regularizer=self.regularizer)
#
#     def call(self, inputs):
#         x, y = inputs
#         c = K.shape(x)[-1]
#         # normalize feature
#         x = tf.nn.l2_normalize(x, axis=1)
#         # normalize weights
#         W = tf.nn.l2_normalize(self.W, axis=0)
#         # dot product
#         logits = x @ W
#         # add margin
#         # clip logits to prevent zero division when backward
#         theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
#         target_logits = tf.cos(self.m * theta)
#         #
#         label_onehot = tf.one_hot(K.cast(y,dtype='int32'), self.n_classes)
#
#         logits = logits * (1 - label_onehot) + target_logits * label_onehot
#         # feature re-scale
#         logits *= self.s
#         out = tf.nn.softmax(logits)
#
#         return out
#
#     def compute_output_shape(self, input_shape):
#         return (None, self.n_classes)


class SphereMargin(Layer):
    def __init__(self, n_classes=10, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0, regularizer=None, **kwargs):
        super(SphereMargin, self).__init__(**kwargs)
        self.n_classes = n_classes
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.regularizer = regularizers.get(regularizer)

        # duplication formula
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def build(self, input_shape):
        super(SphereMargin, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        input, label = inputs

        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # normalize feature
        x = tf.nn.l2_normalize(input, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        cos_theta = x @ W

        cos_m_theta = self.margin_formula[self.m](cos_theta)

        theta = tf.acos(K.clip(cos_m_theta, -1.0 + K.epsilon(), 1.0 - K.epsilon())) #cos_theta.data.acos()
        k = tf.floor((self.m * theta) / math.pi)
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)

        norm_of_feature = tf.norm(input, 2, 1)

        # one_hot = tf.zeros_like(cos_theta)
        # one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (label+1) * phi_theta_ + (1 - (label+1)) * cos_theta
        #output = tf.multiply(output, norm_of_feature)

        output = tf.nn.softmax(output)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)




class CosFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
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
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)