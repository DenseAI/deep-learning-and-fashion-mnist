
# -*- coding:utf-8 -*-

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Embedding, multiply
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model

#from utils.callbacks import CustomCallback, step_decay_schedule

from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os


import numpy as np
import json
import os
import pickle


# Imports.
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist


# VQ layer.
class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                 shape=(self.embedding_dim, self.num_embeddings),
                                 initializer=self.initializer,
                                 trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs ** 2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        # avg_probs = K.mean(encodings, axis=0)
        # perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))

        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)


# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance

        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss

        return recon_loss + loss  # * beta

    return vq_vae_loss



#### CALLBACKS
class CustomCallback(Callback):

    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae

    def on_batch_end(self, batch, logs={}):
        if batch % self.print_every_n_batches == 0:
            # z_new = np.random.normal(size=(1, self.vae.z_dim))
            # reconst = self.vae.decoder.predict(np.array(z_new))[0].squeeze()
            #
            # filepath = os.path.join(self.run_folder,
            #                         'images/img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')
            # if len(reconst.shape) == 2:
            #     plt.imsave(filepath, reconst, cmap='gray_r')
            # else:
            #     plt.imsave(filepath, reconst)
            print("")

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)


class VQVAE():
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_strides,
                 z_dim,
                 data_variance,
                 use_batch_norm=True,
                 use_dropout=True
                 ):

        self.name = 'autoencoder'

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.data_variance = data_variance

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self.learning_rate = 0.0005

        self._build()

    def _build(self):

        # ### THE ENCODER
        # encoder_input = Input(shape=self.input_dim, name='encoder_input')
        #
        # x = encoder_input
        # for i in range(self.n_layers_encoder):
        #     conv_layer = Conv2D(
        #         filters = self.encoder_conv_filters[i],
        #         kernel_size = self.encoder_conv_kernel_size[i],
        #         strides = self.encoder_conv_strides[i],
        #         padding = 'same',
        #         name = 'encoder_conv_' + str(i)
        #     )
        #     x = conv_layer(x)
        #     x = LeakyReLU()(x)
        #
        #     if self.use_batch_norm:
        #         x = BatchNormalization()(x)
        #     if self.use_dropout:
        #         x = Dropout(rate = 0.25)(x)
        #
        # shape_before_flattening = K.int_shape(x)[1:]
        # x = Flatten()(x)
        # encoder_output= Dense(self.z_dim, name='encoder_output')(x)
        #
        # self.encoder = Model(encoder_input, encoder_output)
        #
        #
        # ### THE DECODER
        # decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        #
        # x = Dense(np.prod(shape_before_flattening))(decoder_input)
        # x = Reshape(shape_before_flattening)(x)
        #
        # for i in range(self.n_layers_decoder):
        #     conv_t_layer = Conv2DTranspose(
        #         filters = self.decoder_conv_t_filters[i],
        #         kernel_size = self.decoder_conv_t_kernel_size[i],
        #         strides = self.decoder_conv_t_strides[i],
        #         padding = 'same',
        #         name = 'decoder_conv_t_' + str(i)
        #         )
        #
        #     x = conv_t_layer(x)
        #
        #     if i < self.n_layers_decoder - 1:
        #         x = LeakyReLU()(x)
        #
        #         if self.use_batch_norm:
        #             x = BatchNormalization()(x)
        #
        #         if self.use_dropout:
        #             x = Dropout(rate = 0.25)(x)
        #     else:
        #         x = Activation('sigmoid')(x)
        #
        # decoder_output = x
        #
        # self.decoder = Model(decoder_input, decoder_output)
        #
        # ### THE FULL AUTOENCODER
        # model_input = encoder_input
        # model_output = self.decoder(encoder_output)
        # self.model = Model(model_input, model_output)

        # Encoder
        input_img = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu')(input_img)
        # x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.3)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)

        # VQ-VAE Hyper Parameters.
        embedding_dim = 32  # Length of embedding vectors.
        num_embeddings = 128  # Number of embedding vectors (high value = high bottleneck capacity).
        commitment_cost = 0.25  # Controls the weighting of the loss terms.


        # VQVAELayer.
        enc = Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1), name="pre_vqvae")(x)
        enc_inputs = enc
        enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)
        x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)
        loss = vq_vae_loss_wrapper(self.data_variance, commitment_cost, enc, enc_inputs)

        encoder_output = Flatten()(x)
        self.encoder = Model(input_img, encoder_output)

        # Decoder.
        x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
        x = Conv2DTranspose(1, (3, 3))(x)

        # Autoencoder.
        vqvae = Model(input_img, x)
        vqvae.compile(loss=loss, optimizer='adam')
        vqvae.summary()

        self.model = vqvae


    def compile(self, learning_rate):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate)

        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        self.model.compile(optimizer=optimizer, loss = r_loss)

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
                ], f)

        #self.plot_model(folder)

        


    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    
    def train(self, x_train, x_test, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint2, lr_sched]

        history = self.model.fit(
            x_train,
            x_train,
            batch_size = batch_size,
            shuffle = True,
            epochs = epochs,
            initial_epoch = initial_epoch,
            validation_data=(x_test, x_test),
            callbacks = callbacks_list
        )
        return history

    def plot_model(self, run_folder):
        print("")
        #plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        #plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        #plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)


