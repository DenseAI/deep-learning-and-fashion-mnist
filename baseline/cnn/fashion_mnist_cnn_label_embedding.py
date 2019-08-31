from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Embedding, multiply
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Model

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
import math


from base_utils import plot_confusion_matrix, AdvancedLearnignRateScheduler, get_random_eraser
from networks import create_base_cnn_model_with_kernels_with_label


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



x_train_append = []
y_train_append = []
y_train_random_append = []
num_classes = 10
for ii in range(x_train_with_channels.shape[0]):
    x = x_train_with_channels[ii]
    y = y_train[ii]
    #x_train_append(x)
    for jj in range(num_classes):
        x_train_append.append(x)
        y_train_append.append(y)
        y_train_random_append.append(jj)

x_train_append = np.array(x_train_append)
y_train_append = np.array(y_train_append)
y_train_random_append = np.array(y_train_random_append)

y_train_append_categorical = keras.utils.to_categorical(y_train_append, num_classes)


y_test_rnd = []
for ii in range(len(y_test)):
    label = random.randrange(0, num_classes)
    y_test_rnd.append(label)
y_test_rnd = np.array(y_test_rnd)



# def create_model():
#     learn_rate = 1
#
#     input_label = Input(shape=(1,), dtype='int32')
#     label_embedding = Flatten()(Embedding(num_classes, 128)(input_label))
#
#     # Encoder
#     input_img = Input(shape=(28, 28, 1))
#
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.25)(x)
#
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.25)(x)
#
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#
#     x = multiply([x, label_embedding])
#
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#
#     output = Dense(num_classes, activation='softmax')(x)
#     model = Model([input_img, input_label], output)
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(lr=learn_rate),
#                   metrics=['accuracy'])
#     return model



###################################################################
###  创建模型                                                    ###
###################################################################


kernels = [3]
factor = 4
droprate=0.5
model = create_base_cnn_model_with_kernels_with_label(input_shape, kernels=kernels, optimizer="adamax", droprate=droprate)
model.summary()


model_name = "base_cnn_label_kernel_{}_k_{}_d_{}".format(str(kernels), factor, droprate)
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



def get_random_eraser(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    #def eraser(img):
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
    #return eraser


###################################################################
###  模型训练                                                    ###
###################################################################

load = False
batch_size = 100
epochs = 200
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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, rnd_labels, batch_size=128, dim=(28,28), n_channels=1,
                 n_classes=10, shuffle=True, probability=0.5):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.rnd_labels = rnd_labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.probability = probability
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        y_rnd = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = get_random_eraser(self.data[ID], probability=self.probability)
            # Store class
            y[i] = self.labels[ID]
            y_rnd[i] =  random.randrange(0, num_classes) #self.rnd_labels[ID]
        return [X, y_rnd], keras.utils.to_categorical(y, num_classes=self.n_classes)

# # Generators
training_generator = DataGenerator(x_train_with_channels, y_train, y_train, batch_size=batch_size)
#validation_generator = DataGenerator(x_test_with_channels, y_test,batch_size=batch_size, probability=0.0)




if Training:
    print('Using real-time data augmentation.')
    # Fit the model on the batches generated by datagen.flow().
    model_train_history = model.fit_generator(generator=training_generator,
                                              steps_per_epoch=x_train.shape[0] // batch_size,
                                              validation_data=([x_test_with_channels, y_test_rnd],y_test_categorical),
                                              epochs=epochs,
                                              verbose=2,
                                              workers=4,
                                              callbacks=callbacks)

    for jj in range(len(model_train_history.history['acc'])):
        history_acc.append(model_train_history.history['acc'][jj])
        history_val_acc.append(model_train_history.history['val_acc'][jj])
        history_loss.append(model_train_history.history['loss'][jj])
        history_val_loss.append(model_train_history.history['val_loss'][jj])

if Fine_turning:
    epochs = 20
    model_train_history = model.fit([x_train_append, y_train_random_append], y_train_append_categorical,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=2,
                                    validation_data=([x_test_with_channels, y_test_rnd],y_test_categorical),
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


prediction_classes = model.predict([x_test_with_channels,y_test])
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))



# return ax
filename = './images/{}_confusion_matrix.png'.format(model_name)
# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, filename=filename, normalize=False,
                      title='confusion matrix')
