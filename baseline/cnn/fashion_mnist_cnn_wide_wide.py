from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Embedding, multiply, add
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import math

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

def create_model(factor=1, kernels=[1,3,5,7], droprate=0):
    learn_rate = 1

    input_label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, 128)(input_label))

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
                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),
                  metrics=['accuracy'])
    return model


model = create_model(factor=8, droprate=0.25)
model.summary()



x_train_append = []
y_train_append = []
y_train_random_append = []
num_classes = 10
# for ii in range(x_train_with_channels.shape[0]):
#     x = x_train_with_channels[ii]
#     y = y_train[ii]
#     #x_train_append(x)
#     for jj in range(num_classes):
#         x_train_append.append(x)
#         y_train_append.append(y)
#         y_train_random_append.append(jj)
#
# x_train_append = np.array(x_train_append)
# y_train_append = np.array(y_train_append)
# y_train_random_append = np.array(y_train_random_append)
#
# y_train_append_categorical = keras.utils.to_categorical(y_train_append, num_classes)


y_test_rnd = []
for ii in range(len(y_test)):
    label = random.randrange(0, num_classes)
    y_test_rnd.append(label)
y_test_rnd = np.array(y_test_rnd)


checkpoint_path = './weights/cnn_wide_weight.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback =  ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 save_weights_only=True,
                                 period=1) #  save weights every 1 epochs

batch_size = 128
epochs = 30


load = True

if load:
	model.load_weights(checkpoint_path)


def get_random_eraser2(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    def eraser(img):
        if random.uniform(0, 1) > probability:
            return img

        img_h, img_w, img_c = img.shape
        for attempt in range(100):
            area = img_h * img_w

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)
                if img_c == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                return img

        return img
    return eraser


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_random_eraser2(probability = 0.33))


datagen.fit(x_train_with_channels)

DATA_GEN = True
if DATA_GEN :
    model_train_history = model.fit_generator(datagen.flow(x_train_with_channels, y_train_categorical, batch_size=batch_size),
                                              steps_per_epoch=len(x_train_with_channels) / batch_size,
                                              epochs=epochs,
                                              validation_data=(x_test_with_channels, y_test_categorical),
                                              workers=4,
                                              callbacks=[cp_callback])

else:
    model_train_history = model.fit(x_train_with_channels, y_train_categorical,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test_with_channels, y_test_categorical),
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
plt.savefig('./images/cnn_wide_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/cnn_wide_loss.png')
plt.show()


prediction_classes = model.predict(x_test_with_channels)
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout();
    plt.savefig('./images/cnn_wide_confusion_matrix.png')


# return ax

# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,
                      title='confusion matrix')
