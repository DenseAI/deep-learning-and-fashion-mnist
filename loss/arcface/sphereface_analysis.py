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
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
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

def create_model():
    learn_rate = 1

    # Encoder
    input_img = Input(shape=(28, 28, 1))
    input_label = Input(shape=(1,))
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
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)


    output = x  #SphereFace(num_classes, m=m)([x, input_label])
    model = Model(input_img, output)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=learn_rate),
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()

loss_name = "sphereface"

model.load_weights(os.path.join(RUN_FOLDER, 'weights/{}-weights.ckpt'.format(loss_name)))

x_test_append = []
y_test_append = []
y_test_rnd_append = []
for ii in range(len(y_test)):
    for jj in range(num_classes):
        label = random.randrange(0, num_classes)
        x_test_append.append(x_test[ii])
        y_test_append.append(y_test[ii])
        y_test_rnd_append.append(jj)

x_test_append = np.array(x_test_append)
y_test_append = np.array(y_test_append)
y_test_rnd_append = np.array(y_test_rnd_append)

prediction_classes = model.predict([x_test_append, y_test_rnd_append])

print(prediction_classes[0:num_classes])


y_preds = []
preds = np.zeros(num_classes)
for ii in range(len(prediction_classes)):
    preds = preds + prediction_classes[ii]
    if(ii > 0 and ii % num_classes == 0):
        y_preds.append(preds)
        preds = np.zeros(num_classes)
    elif(ii == (len(prediction_classes) - 1)):
        y_preds.append(preds)

y_preds = np.array(y_preds)
#print(y_test[0])
print(y_preds.shape)

y_preds_raw = y_preds

y_preds = np.argmax(y_preds, axis=1)
print(classification_report(y_test, y_preds))


for ii in range(len(y_test)):
    if(y_test[ii] != y_preds[ii] and y_test[ii] == 0):
        print(y_test[ii], " ", y_preds[ii])
        print(y_preds_raw[ii])
        print(prediction_classes[ii * 10])
        print(prediction_classes[ii * 10+1])
        print(prediction_classes[ii * 10+2])
        print(prediction_classes[ii * 10+3])
        print(prediction_classes[ii * 10+4])
        print(prediction_classes[ii * 10+5])
        print(prediction_classes[ii * 10+6])
        print(prediction_classes[ii * 10+7])
        print(prediction_classes[ii * 10+8])
        print(prediction_classes[ii * 10+9])

    # if(y_preds_raw[ii][0] > 1):
    #     print(y_test[ii], " ", y_preds[ii])
    #     print(y_preds_raw[ii])




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
    plt.savefig('./images/ae_confusion_matrix.png')


# return ax

# y_preds = np.array(y_preds)
#
classes = ["Zero","Top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

print(y_preds)
# Plot confusion matrix
plot_confusion_matrix(y_test, y_preds, classes=classes, normalize=False,
                      title='confusion matrix')
#
#
#
# print(classification_report(example_labels, y_preds))

#
# sphereface_model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
# sphereface_model.summary()
#
#
# sphereface_features = sphereface_model.predict(x_test, verbose=1)
# sphereface_features /= np.linalg.norm(sphereface_features, axis=1, keepdims=True)
#
# # plot
# fig4 = plt.figure()
# ax4 = Axes3D(fig4)
# for c in range(len(np.unique(y_test))):
#     ax4.plot(sphereface_features[y_test==c, 0], sphereface_features[y_test==c, 1], sphereface_features[y_test==c, 2], '.', alpha=0.1)
# plt.title('SphereFace')
#
# plt.show()
#
#
