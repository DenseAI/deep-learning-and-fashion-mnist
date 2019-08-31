import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import pickle
from keras.datasets import mnist, fashion_mnist

#from utils.loaders import load_mnist, load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras import regularizers
from mpl_toolkits.mplot3d import Axes3D
import random

from ArcFace import ArcFace, SphereFace, CosFace

def load_model(model_class, folder):
    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)
    model.load_weights(os.path.join(folder, 'weights/weights.ckpt'))
    return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data() #mnist.load_data()

    # x_train_zero = np.zeros((6000, 28, 28), dtype="float32")
    # y_train_zero = np.zeros((6000), dtype="float32")
    #
    # x_test_zero = np.zeros((1000, 28, 28), dtype="float32")
    # y_test_zero = np.zeros((1000), dtype="float32")
    #
    # y_train = y_train + 1
    # y_test = y_test + 1
    #
    # x_train = np.vstack([x_train, x_train_zero])
    # x_test = np.vstack([x_test, x_test_zero])
    # y_train = np.hstack([y_train, y_train_zero])
    # y_test = np.hstack([y_test, y_test_zero])


    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    return (x_train, y_train), (x_test, y_test)



# run params
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = './'
#RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

(x_train, y_train), (x_test, y_test) = load_mnist()

num_classes = 10
m = 2

def create_model(kernels=[3,3],droprate=0.25, factor=1):
    learn_rate = 1

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


    output = x  #SphereFace(num_classes, m=m)([x, input_label])
    model = Model(input_img, output)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()

loss_name = "sphereface"

model.load_weights(os.path.join(RUN_FOLDER, 'weights/{}-weights.ckpt'.format(loss_name)), by_name=True)


n_to_show = 1000
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]


labels = []
for jj in range(n_to_show):
    labels.append(0)
labels = np.array(labels)

x_train_z_points = model.predict(x_train)
x_test_z_points = model.predict(example_images)

z_point_dict = {}
z_point_norm_dict = {}
for ii in range(1):
    labels = []
    for jj in range(n_to_show):
        labels.append(ii)
    labels = np.array(labels)

    #image = example_images[index]
    #example_images = np.reshape(example_images, (1, example_images.shape[0], image.shape[1], image.shape[2]))

    x_test_z_points = model.predict(example_images)
    z_points = x_test_z_points
    z_point_dict[ii] = z_points
    test_norms = []
    for jj in range(len(z_points)):
        aa = z_points[jj]
        norm = 0.0
        for kk in range(len(aa)):
            norm += aa[kk] ** 2

        if norm != 0.0:
            norm = norm ** 0.5
        test_norms.append(norm)
    test_norms = np.array(test_norms)
    z_point_norm_dict[ii] = test_norms

print(z_point_dict.keys())
print(z_point_norm_dict.keys())

train_norms = []
for ii in range(len(x_train_z_points)):
    aa = x_train_z_points[ii]
    norm = 0.0
    for jj in range(len(aa)):
        norm += aa[jj] ** 2

    if norm != 0.0 :
        norm = norm ** 0.5
    train_norms.append(norm)
train_norms = np.array(train_norms)

error_dict = {}
counter = 0
error = 0


error_dict = {}
counter = 0
error = 0

y_preds = []
for ii in range(len(x_test_z_points)):

    # aa = x_test_z_points[ii]
    # normA = 0.0
    # for jj in range(len(aa)):
    #     normA += aa[jj] ** 2
    #
    # if normA != 0.0:
    #     normA = normA ** 0.5

    max = -1000000
    index = -1
    sim_all = []
    for jj in range(len(x_train_z_points)):

        label_index = y_train[jj]
        aa = z_point_dict[0][ii]

        normA = z_point_norm_dict[0][ii]

        bb = x_train_z_points[jj]
        dot_product = np.dot(aa, bb)
        normB = train_norms[jj]

        sim = 0.0
        if normA == 0.0 or normB == 0.0:
            sim = 0.0
        else:
            sim = dot_product / ((normA) * (normB))

        if sim > max and sim < 0.9999:
            max = sim
            index = jj

        sim_all.append(sim)

    y_preds.append(y_train[index])

    if index > -1 and ( example_labels[ii] == y_train[index]):
        counter = counter + 1
    else:
        error = error + 1
        sim_all = np.array(sim_all)
        sorted_dela_errors = np.argsort(sim_all)
        most_important_errors = sorted_dela_errors[-15:]
        print("error: ", error, y_train[most_important_errors], sim_all[most_important_errors])
        for kk in range(len(most_important_errors)):
            if y_train[most_important_errors[kk]] != example_labels[ii]:
                if most_important_errors[kk] in error_dict.keys():
                    error_dict[most_important_errors[kk]] = error_dict[most_important_errors[kk]] + 1
                else:
                    error_dict[most_important_errors[kk]] = 1

    print(ii, " ", example_labels[ii], " ", y_train[index], " ", max)

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
    plt.savefig('./images/cae_confusion_matrix.png')


# return ax

y_preds = np.array(y_preds)

classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

print(example_labels)
print(y_preds)
# Plot confusion matrix
plot_confusion_matrix(example_labels, y_preds, classes=classes, normalize=False,
                      title='confusion matrix')



print(classification_report(example_labels, y_preds))
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
    plt.savefig('./images/cae_confusion_matrix.png')


# return ax

y_preds = np.array(y_preds)

classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

print(example_labels)
print(y_preds)
# Plot confusion matrix
plot_confusion_matrix(example_labels, y_preds, classes=classes, normalize=False,
                      title='confusion matrix')



print(classification_report(example_labels, y_preds))