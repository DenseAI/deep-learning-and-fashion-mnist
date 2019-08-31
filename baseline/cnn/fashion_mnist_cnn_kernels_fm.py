
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
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract, multiply
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, add, Embedding, multiply, subtract, add, dot, Dot
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

from base_utils import plot_confusion_matrix, AdvancedLearnignRateScheduler, get_random_eraser




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
    # x = BatchNormalization()(x)
    # if droprate > 0:
    #     x = Dropout(droprate)(x)

    output =  x #Dense(num_classes, activation="softmax")(x)
    model = Model(input_img, output)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


k_latent = 10
embedding_reg = 0.0002
kernel_reg = 0.1

def get_embed(x_input, x_size, k_latent):
    if x_size > 0:  # category
        embed = Embedding(x_size, k_latent, input_length=1,
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed

def build_fm_model(f_size):
    dim_input = len(f_size)

    input_x = [Input(shape=(1,)) for i in range(dim_input)]

    biases = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]

    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]

    s = Add()(factors)

    diffs = [Subtract()([s, x]) for x in factors]

    dots = [Dot(axes=1)([d, x]) for d ,x in zip(diffs, factors)]

    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    #opt = Adam(clipnorm=0.5)
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    output_f = factors + biases
    #model_features = Model(inputs=input_x, outputs=output_f)
    return model  #, model_features



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

fm_checkpoint_path = './weights/{}_fm_weight.ckpt'.format(model_name)
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    # Early stopping definition
    #EarlyStopping(monitor=loss_value, patience=20, verbose=1),
    # Decrease learning rate by 0.1 factor
    AdvancedLearnignRateScheduler(monitor=loss_value, patience=10, verbose=1, mode='auto', decayRatio=0.9),
    # Saving best model
    ModelCheckpoint(fm_checkpoint_path, monitor=loss_value, save_best_only=True, verbose=1),
]



###################################################################
###  模型训练                                                    ###
###################################################################

load = True
batch_size = 100
epochs = 50
data_augmentation = True
pixel_level = True

Training = False
Fine_turning = True

if load:
    model.load_weights(checkpoint_path, by_name=True)

FEATURES = 128
feature_size = []
for ii in range(FEATURES):
    feature_size.append(0)


fm_model = build_fm_model(feature_size)
fm_model.summary()

train_pred = model.predict(x_train_with_channels)
test_pred = model.predict(x_test_with_channels)


print("train_pred: ", train_pred.shape)
print("test_pred: ", test_pred.shape)

x_train_fm = []
x_test_fm = []
#for ii in range(len(train_pred)):
for jj in range(FEATURES):
    x_train_fm.append(train_pred[:, jj:(jj+1)])
    x_test_fm.append(test_pred[:, jj:(jj + 1)])



history_acc = []
history_val_acc = []
history_loss = []
history_val_loss = []

# if Training:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False,  # randomly flip images
#         preprocessing_function=get_random_eraser(probability = 0.5))
#
#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train_with_channels)
#
#     # Fit the model on the batches generated by datagen.flow().
#     model_train_history = model.fit_generator(datagen.flow(x_train_with_channels, y_train_categorical),
#                                               steps_per_epoch=x_train_with_channels.shape[0] // batch_size,
#                                               validation_data=(x_test_with_channels, y_test_categorical),
#                                               epochs=epochs,
#                                               verbose=1,
#                                               workers=4,
#                                               callbacks=callbacks)
#
#     for jj in range(len(model_train_history.history['acc'])):
#         history_acc.append(model_train_history.history['acc'][jj])
#         history_val_acc.append(model_train_history.history['val_acc'][jj])
#         history_loss.append(model_train_history.history['loss'][jj])
#         history_val_loss.append(model_train_history.history['val_loss'][jj])

if Fine_turning:
    epochs = 30
    model_train_history = fm_model.fit(x_train_fm, y_train_categorical,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test_fm, y_test_categorical),
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


prediction_classes = fm_model.predict(x_test_fm)
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))



# return ax
filename = './images/{}_confusion_matrix.png'.format(model_name)
# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, filename=filename, normalize=False,
                      title='confusion matrix')
