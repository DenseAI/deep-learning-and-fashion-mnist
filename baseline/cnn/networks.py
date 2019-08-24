


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, add
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

def create_base_cnn_model(input_shape, num_classes=10):
    learn_rate = 1

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding = 'same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),
                  metrics=['accuracy'])
    return model


def create_base_cnn_model_with_optimizer(input_shape, optimizer="sgd", num_classes=10):
    learn_rate = 1

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding = 'same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def create_base_cnn_model_with_kernel(input_shape, kernel=3, optimizer="sgd", num_classes=10):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(kernel, kernel), activation='relu', input_shape=input_shape, padding = 'same'))
    model.add(Conv2D(32, (kernel, kernel), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def create_base_cnn_model_with_kernels(input_shape, kernels=[3,3,3,3], optimizer="adamax", droprate=0.25, factor=1, num_classes=10):
    # Encoder
    input_img = Input(shape=(28, 28, 1))

    outs = []
    for kernel in kernels:
        x = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(input_img)
        x = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        if droprate > 0:
            x = Dropout(droprate)(x)

        x = Conv2D(64 * factor, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64 * factor, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        if droprate > 0:
            x = Dropout(droprate)(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outs.append(x)

    x = add(outs)
    x = BatchNormalization()(x)

    if droprate > 0:
        x = Dropout(droprate)(x)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(input_img, output)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model