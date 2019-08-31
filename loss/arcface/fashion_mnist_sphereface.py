from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, add
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras import regularizers


# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Model

from ArcFace import ArcFace, SphereFace, CosFace


weight_decay = 1e-4

###################################################################
###  配置 Tensorflow                                            ###
###################################################################
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


###################################################################
###  读取训练、测试数据                                           ###
###################################################################
num_classes = 10

# image dimensions
img_rows, img_cols = 28, 28

classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


def load_data_from_keras():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # x_train_zero = np.zeros((6000*2, 28,28), dtype="float32")
    # y_train_zero = np.zeros((6000), dtype="float32")
    # y_train_one = np.ones((6000), dtype="float32")
    #
    # x_test_zero = np.zeros((1000*2, 28, 28), dtype="float32")
    # y_test_zero = np.zeros((1000), dtype="float32")
    # y_test_one = np.ones((1000), dtype="float32")
    #
    # y_train = y_train + 2
    # y_test = y_test + 2

    # x_train = np.vstack([x_train, x_train_zero])
    # x_test = np.vstack([x_test, x_test_zero])
    # y_train = np.hstack([y_train, y_train_zero, y_train_one])
    # y_test = np.hstack([y_test, y_test_zero, y_test_one])

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data_from_keras()


if K.image_data_format() == 'channels_first':
    x_train_with_channels = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test_with_channels = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_with_channels = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test_with_channels = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("train feature shape = ", x_train_with_channels.shape)
print("test feature shape = ", x_test_with_channels.shape)


x_train_with_channels = x_train_with_channels.astype("float32") / 255.0
x_test_with_channels = x_test_with_channels.astype("float32") / 255.0

y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)


y_test_rnd = []
for ii in range(len(y_test)):
    label = random.randrange(0, num_classes)
    y_test_rnd.append(label)
y_test_rnd = np.array(y_test_rnd)



###################################################################
###  创建模型                                                    ###
###################################################################

loss_name = "sphereface"
m=2

def create_model(kernels=[3],droprate=0.25, factor=1):
    input_img = Input(shape=(28, 28, 1))
    input_label = Input(shape=(1,))

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

    if(len(outs) > 1):
        x = add(outs)
    else:
        x = outs[0]

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = SphereFace(num_classes, m=m, regularizer=regularizers.l2(weight_decay))([x, input_label])
    model = Model([input_img, input_label], output)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

kernels = [3,3,3,3]
factor = 4

model = create_model()
model.summary()


###################################################################
###  模型训练                                                    ###
###################################################################


model_name = "sphereface_kernel_{}_k_{}".format(str(kernels), factor)
loss_value = 'val_acc'
checkpoint_path = './weights/{}_weight.ckpt'.format(model_name)
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    # Early stopping definition
    #EarlyStopping(monitor=loss_value, patience=20, verbose=1),
    # Decrease learning rate by 0.1 factor
    #AdvancedLearnignRateScheduler(monitor=loss_value, patience=10, verbose=1, mode='auto', decayRatio=0.9),
    # Saving best model
    ModelCheckpoint(checkpoint_path, monitor=loss_value, save_best_only=True, verbose=1),
]

batch_size = 128
epochs = 50
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")
model_train_history = model.fit([x_train_with_channels, (y_train)], y_train_categorical,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=([x_test_with_channels,(y_test)], y_test_categorical),
                                callbacks=callbacks)


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
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_loss_{}.png'.format(loss_name, m))
plt.show()


prediction_classes = model.predict([x_test_with_channels,y_test])
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
    plt.savefig('./images/{}_confusion_matrix_{}.png'.format(loss_name, m))


# return ax

# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,
                      title='confusion matrix')



