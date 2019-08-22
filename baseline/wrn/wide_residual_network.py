from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, merge, Concatenate, Conv2D, Embedding, multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from keras.regularizers import l2
from keras import backend as K

import logging
logging.basicConfig(level=logging.DEBUG)


depth = 28              # table 5 on page 8 indicates best value (4.17) CIFAR-10
k = 10                  # 'widen_factor'; table 5 on page 8 indicates best value (4.17) CIFAR-10
dropout_probability = 0 # table 6 on page 10 indicates best value (4.17) CIFAR-10

weight_decay = 0.0005   # page 10: "Used in all experiments"

# Other config from code; throughtout all layer:
use_bias = True        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua


# Keras specific
if K.image_dim_ordering() == "th":
    #logging.debug("image_dim_ordering = 'th'")
    channel_axis = 1
else:
    #logging.debug("image_dim_ordering = 'tf'")
    channel_axis = -1

# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride, dropoutRate):
    def f(net):
        equalInOut = (n_input_plane == n_output_plane)
        n_bottleneck_plane = n_output_plane

        # Residual block
        if not equalInOut:
            net = BatchNormalization(axis=channel_axis)(net)
            net = Activation("relu")(net)
        else:
            convs = BatchNormalization(axis=channel_axis)(net)
            convs = Activation("relu")(convs)

        convs = Conv2D(n_bottleneck_plane,   #conv1
                       3,
                       strides=stride,
                       padding="same",
                       init=weight_init,
                       W_regularizer=l2(weight_decay),
                       use_bias=use_bias)(convs if equalInOut else net)
        convs = BatchNormalization(axis=channel_axis)(convs)
        convs = Activation("relu")(convs)
        if dropoutRate > 0:
            convs = Dropout(dropoutRate)(convs)
        convs = Conv2D(n_bottleneck_plane,   #conv2
                       3,
                       strides=1,
                       padding="same",
                       init=weight_init,
                       W_regularizer=l2(weight_decay),
                       use_bias=use_bias)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(n_output_plane,  #convShortcut
                              1,
                              strides=stride,
                              padding="same",
                              init=weight_init,
                              W_regularizer=l2(weight_decay),
                              use_bias=use_bias)(net)
        else:
            shortcut = net

        return Add()([convs, shortcut])

    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride, dropoutRate):
    def f(net):
        for i in range(int(count)):
            net = block(i == 0 and n_input_plane or n_output_plane, n_output_plane, i == 0 and stride or (1, 1), dropoutRate = dropoutRate)(net)
        return net

    return f

def create_wide_residual_network(input_shape, depth=28, nb_classes=10, k=2, dropoutRate=0.0):
    logging.debug("Creating model...")

    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    inputs = Input(shape=input_shape)

    n_stages = [16, 16 * k, 32 * k, 64 * k]

    conv1 = Conv2D(nb_filter=n_stages[0],
                   kernel_size = 3,
                   strides=1,
                   padding="same",
                   init=weight_init,
                   W_regularizer=l2(weight_decay),
                   use_bias=use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=1, dropoutRate=dropoutRate)(
        conv1)  # "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=2, dropoutRate=dropoutRate)(
        conv2)  # "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=2, dropoutRate=dropoutRate)(
        conv3)  # "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm)

    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
    x = Flatten()(pool)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(output_dim=nb_classes, init=weight_init, bias=use_bias,
                        W_regularizer=l2(weight_decay), activation="softmax")(x)

    model = Model(input=inputs, output=predictions)
    return model


def create_wide_residual_network_with_label(input_shape, depth=28, nb_classes=10, k=2, dropoutRate=0.0):
    logging.debug("Creating model...")

    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    input_label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(nb_classes, 128)(input_label))

    input_img = Input(shape=input_shape)

    n_stages = [16, 16 * k, 32 * k, 64 * k]

    conv1 = Conv2D(nb_filter=n_stages[0],
                   kernel_size = (3,3),
                   strides=(1, 1),
                   padding="same",
                   init=weight_init,
                   W_regularizer=l2(weight_decay),
                   use_bias=use_bias)(input_img)  # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1), dropoutRate=dropoutRate)(
        conv1)  # "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2), dropoutRate=dropoutRate)(
        conv2)  # "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2), dropoutRate=dropoutRate)(
        conv3)  # "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm)

    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
    x = Flatten()(pool)
    x = Dense(128, activation='relu')(x)
    x = multiply([x, label_embedding])
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim=nb_classes, init=weight_init, bias=use_bias,
                        W_regularizer=l2(weight_decay), activation="softmax")(x)

    model = Model(input=[input_img, input_label], output=predictions)


    return model



if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model

    i = 0
    in_planes = 16
    out_planes = 32
    print(i == 0 and in_planes or out_planes)

    i = 1

    print(i == 0 and in_planes or out_planes)

    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, depth=16, nb_classes=10, k=2, dropoutRate=0.25)

    wrn_28_10.summary()
