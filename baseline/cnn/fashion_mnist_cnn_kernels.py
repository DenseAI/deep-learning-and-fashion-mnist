
# -*- coding:utf-8 -*-


from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from keras.models import Model

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

from base_utils import plot_confusion_matrix, AdvancedLearnignRateScheduler, get_random_eraser
from networks import create_base_cnn_model_with_kernels



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




###################################################################
###  创建模型                                                    ###
###################################################################

#optimizers = SGD、RMSprop、Adagrad、Adadelta、Adam、Adamax、Nadam
# sgd = SGD
# rmsprop = RMSprop
# adagrad = Adagrad
# adadelta = Adadelta
# adam = Adam
# adamax = Adamax
# nadam = Nadam

kernels = [3,3,3,3]
factor = 4
model = create_base_cnn_model_with_kernels(input_shape, kernels=kernels, factor=4, optimizer="sgd")
model.summary()


model_name = "base_cnn_kernel_{}_k_{}".format(str(kernels), factor)
loss_value = 'val_acc'
checkpoint_path = './weights/{}_weight.ckpt'.format(model_name)
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    # Early stopping definition
    #EarlyStopping(monitor=loss_value, patience=20, verbose=1),
    # Decrease learning rate by 0.1 factor
    AdvancedLearnignRateScheduler(monitor=loss_value, patience=10, verbose=1, mode='auto', decayRatio=0.9),
    # Saving best model
    ModelCheckpoint(checkpoint_path, monitor=loss_value, save_best_only=True, verbose=1),
]



###################################################################
###  模型训练                                                    ###
###################################################################

load = True
batch_size = 100
epochs = 100
data_augmentation = True
pixel_level = True

Training = True
Fine_turning = False

if load:
    model.load_weights(checkpoint_path)


history_acc = []
history_val_acc = []
history_loss = []
history_val_loss = []

if Training:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_random_eraser(probability = 0.5))

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train_with_channels)

    # Fit the model on the batches generated by datagen.flow().
    model_train_history = model.fit_generator(datagen.flow(x_train_with_channels, y_train_categorical),
                                              steps_per_epoch=x_train_with_channels.shape[0] // batch_size,
                                              validation_data=(x_test_with_channels, y_test_categorical),
                                              epochs=epochs,
                                              verbose=1,
                                              workers=4,
                                              callbacks=callbacks)

    for jj in range(len(model_train_history.history['acc'])):
        history_acc.append(model_train_history.history['acc'][jj])
        history_val_acc.append(model_train_history.history['val_acc'][jj])
        history_loss.append(model_train_history.history['loss'][jj])
        history_val_loss.append(model_train_history.history['val_loss'][jj])

if Fine_turning:
    epochs = 30
    model_train_history = model.fit(x_train_with_channels, y_train_categorical,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test_with_channels, y_test_categorical),
                                    callbacks=callbacks)
    for jj in range(len(model_train_history.history['acc'])):
        history_acc.append(model_train_history.history['acc'][jj])
        history_val_acc.append(model_train_history.history['val_acc'][jj])
        history_loss.append(model_train_history.history['loss'][jj])
        history_val_loss.append(model_train_history.history['val_loss'][jj])

###################################################################
###  保存训练信息                                                ###
###################################################################

print(history_acc)
print(history_val_acc)
print(history_loss)
print(history_val_loss)


# Save
filename = "{}_result.npz".format(model_name)
save_dict = {
    "acc": history_acc,
    "val_acc": history_val_acc,
    "loss": history_loss,
    "val_loss":history_val_loss
}
output = os.path.join("./results/", filename)
np.savez(output, **save_dict)

# Plot training & validation accuracy values
plt.plot(np.array(history_acc))
plt.plot(np.array(history_val_acc))
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_acc.png'.format(model_name))
plt.show()

# Plot training & validation loss values
plt.plot(np.array(history_loss))
plt.plot(np.array(history_val_loss))
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_loss.png'.format(model_name))
plt.show()


prediction_classes = model.predict(x_test_with_channels)
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))



# return ax
filename = './images/{}_confusion_matrix.png'.format(model_name)
# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, filename=filename, normalize=False,
                      title='confusion matrix')
