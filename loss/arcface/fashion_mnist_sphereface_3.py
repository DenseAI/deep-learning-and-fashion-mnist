from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, InputSpec
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras import regularizers
from keras import initializers, constraints

from keras.layers import Layer

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Model

from ArcFace import ArcFace, SphereFace, CosFace
import math

weight_decay = 1e-4

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

num_classes = 10

# image dimensions
img_rows, img_cols = 28, 28

classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


def load_data_from_keras():
    # get data using tf.keras.datasets. Train and test set is automatically split from datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data_from_keras()



#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

if K.image_data_format() == 'channels_first':
    x_train_with_channels = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #x_val_with_channels = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    x_test_with_channels = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_with_channels = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_val_with_channels = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    x_test_with_channels = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print("train feature shape = ", x_train_with_channels.shape)
#print("validation feature shape = ", x_val_with_channels.shape)
print("test feature shape = ", x_test_with_channels.shape)


x_train_with_channels = x_train_with_channels.astype("float32") / 255.0
#x_val_with_channels = x_val_with_channels.astype("float32") / 255.0
x_test_with_channels = x_test_with_channels.astype("float32") / 255.0

y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
#y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)


loss_name = "sphereface"
m = 2

s = 30.0


def softmax_loss(y_true, y_pred, s = 30.0, e=0.1):
    logits = y_pred
    print(logits)
    print(y_pred)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    return loss

def sphereface_loss(y_true, y_pred, batch_size=128, input_dim=128, output_dim=10, s = 30.0, e=0.1, l=0.0):
    labels = y_true
    #embeddings_norm = y_pred[0:input_dim]
    #orgina_logits = y_pred[1]

    #logits, target_logits
    phi_theta_ = y_pred[:, 0:output_dim]
    cos_theta = y_pred[:, output_dim:]
    # cosine = y_pred[:, input_dim:]
    #
    print("phi: ", phi_theta_)
    print("cosine: ", cos_theta)
    #
    # output = (labels * phi) + ((1.0 - labels) * cosine)
    # output *= s

    #logits =    y_true* logits + (1 - y_true) * target_logits
    #logits *= 30.0 #norm_of_feature

    #output = (1 - labels) * phi_theta_ +  labels * cos_theta

    #loss = tf.nn.softmax(output)

    #label_onehot = tf.one_hot(y_true, nrof_classes)
    adjust_theta = s * tf.where(tf.equal(y_true, 1), phi_theta_, cos_theta)

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=adjust_theta, labels=y_true))
    loss = tf.nn.softmax(adjust_theta)
    return loss



class SpherefaceLayer(Layer):
    def __init__(self, units,
                 use_bias=True,
                 m = 4,
                 easy_margin = False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SpherefaceLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.m = m
        self.easy_margin = easy_margin
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        l = 0.

        embeddings = inputs[0]
        labels = input[1]
        embeddings_norm = tf.norm(embeddings, axis=1)

        #with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                  shape=[embeddings.get_shape().as_list()[-1], 10],
                                  initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.nn.l2_normalize(weights, axis=0)
        # cacualting the cos value of angles between embeddings and weights
        orgina_logits = tf.matmul(embeddings, weights)
        N = embeddings.get_shape()[0]  # get batch_size
        single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
        # N = 128, labels = [1,0,...,9]
        # single_sample_label_index:
        # [ [0,1],
        #   [1,0],
        #   ....
        #   [128,9]]
        selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index)
        cos_theta = tf.div(selected_logits, embeddings_norm)
        cos_theta_power = tf.square(cos_theta)
        cos_theta_biq = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2 * cos_theta_power - 1), sign0)
        sign4 = 2 * sign0 + sign3 - 3
        result = sign3 * (8 * cos_theta_biq - 8 * cos_theta_power + 1) + sign4

        margin_logits = tf.multiply(result, embeddings_norm)
        f = 1.0 / (1.0 + l)
        ff = 1.0 - f
        combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                              tf.subtract(margin_logits, selected_logits),
                                                              orgina_logits.get_shape()))
        updated_logits = ff * orgina_logits + f * combined_logits
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
        pred_prob = tf.nn.softmax(logits=updated_logits)
        #return pred_prob, loss

        return updated_logits

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units  #+ input_shape[-1]
        return tuple(output_shape)


class MyLayer(Layer):

    def __init__(self, output_dim, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0, **kwargs):
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'

        self.output_dim = output_dim
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        # duplication formula
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):

        x, label = inputs
        # self.iter += 1
        # self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
        #
        # x = tf.nn.l2_normalize(x, axis=1)
        # # normalize weights
        # W = tf.nn.l2_normalize(self.kernel, axis=0)
        #
        # cos_theta = K.dot(x, W)
        # #cos_theta = cos_theta(-1, 1)
        #
        # cos_m_theta = self.margin_formula[self.m](cos_theta)
        # theta = tf.acos(cos_theta)
        # k = (self.m * theta) / math.pi
        # phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        # phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        # norm_of_feature = tf.norm(x, 2, 1)

        # kernel = tf.get_variable(name='kernel',dtype=tf.float32,shape=[args.embedding_size,nrof_classes],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        # cos_theta = tf.matmul(embeddings, kernel_norm)
        # cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
        # phi = cos_theta - m
        # label_onehot = tf.one_hot(label_batch, nrof_classes)
        # adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)

        kernel_norm = tf.nn.l2_normalize(self.kernel, axis=0)
        cos_theta = tf.matmul(x, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)

        phi = cos_theta - m


        #one_hot = tf.zeros_like(cos_theta)
        #one_hot.scatter_(1, label, 1)

        #output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        #output *= 30.0 #norm_of_feature
        #
        # print(phi_theta_)
        # print(cos_theta)

        # x, y = inputs
        # c = K.shape(x)[-1]
        # # normalize feature
        # x = tf.nn.l2_normalize(x, axis=1)
        # # normalize weights
        # W = tf.nn.l2_normalize(self.kernel, axis=0)
        # # dot product
        # logits = x @ W
        # # add margin
        # # clip logits to prevent zero division when backward
        # theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        # target_logits = tf.cos(self.m * theta)
        #
        #logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        #logits *= self.s

        # print(logits)
        # print(target_logits)

        return K.concatenate([phi, cos_theta])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], self.output_dim * 2)

nb_classes = 10
def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2


def create_model():
    learn_rate = 1

    # Encoder
    input_img = Input(shape=(28, 28, 1))
    input_label = Input(shape=(10,))
    x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = MyLayer(10)([x, input_label])
    model = Model([input_img, input_label], output)

    model.compile(loss=sphereface_loss,
                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()


checkpoint_path = './weights/{}-weights.ckpt'.format(loss_name)
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback =  ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 save_weights_only=True,
                                 period=1) #  save weights every 1 epochs




batch_size = 128
epochs = 50
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")
model_train_history = model.fit([x_train_with_channels, y_train_categorical] , y_train_categorical,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=([x_test_with_channels, y_test_categorical], y_test_categorical),
                                callbacks=[cp_callback])


print(model_train_history.history['acc'])
print(model_train_history.history['val_acc'])
print(model_train_history.history['loss'])
print(model_train_history.history['val_loss'])

# Plot training & validation accuracy values
plt.plot(model_train_history.history['acc'])
plt.plot(model_train_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_acc_{}.png'.format(loss_name, m))
#plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_loss_{}.png'.format(loss_name, m))
#plt.show()

#
# prediction_classes = model.predict([x_test_with_channels,y_test])
# prediction_classes = np.argmax(prediction_classes, axis=1)
# print(classification_report(y_test, prediction_classes))
#
#
# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     # classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout();
#     plt.savefig('./images/{}_confusion_matrix_{}.png'.format(loss_name, m))
#
#
# # return ax
#
# # Plot confusion matrix
# plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,
#                       title='confusion matrix')
