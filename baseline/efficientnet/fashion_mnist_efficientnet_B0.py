# -*- coding:utf-8 -*-

#原作者：苏剑林 http://kexue.fm/archives/4556/

import numpy as np

from tqdm import tqdm
from scipy import misc
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt
import efficientnet as efn
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from base_utils import plot_confusion_matrix, AdvancedLearnignRateScheduler, get_random_eraser


###################################################################
###  读取训练、测试数据                                           ###
###################################################################
np.random.seed(2017)
tf.set_random_seed(2017)

classes = ["Top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

#加载数据
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#图片进行高、宽调整，因为Mobilenet最小只支持32*32
height,width = 56,56

input_image = Input(shape=(height,width))
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)

#mobilenet模型
#base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')

X_train = X_train.reshape((-1,28,28))
X_train = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_train))])/255.

X_test = X_test.reshape((-1,28,28))
X_test = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_test))])/255.


print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

img_rows, img_cols = 56, 56
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #x_val_with_channels = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #x_val_with_channels = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



###################################################################
###  创建模型                                                    ###
###################################################################
num_classes = 10


base_model = efn.model.EfficientNetB0(input_shape=input_shape, classes=num_classes)

# output = Dropout(0.25)(base_model.output)
# predict = Dense(10, activation='softmax')(output)

model = base_model #Model(inputs=input_image, outputs=predict)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()



model_name = "efficientnet_b0"
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


if load:
    model.load_weights(checkpoint_path)


if not data_augmentation:
    model_train_history = model.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=2,
                                    validation_data=(X_test, y_test),
                                    callbacks=callbacks)

else:
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
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_random_eraser(probability = 0.5))

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model_train_history = model.fit_generator(datagen.flow(X_train, y_train),
                                              steps_per_epoch=X_train.shape[0] // batch_size,
                                              validation_data=(X_test, y_test),
                                              epochs=epochs,
                                              verbose=2,
                                              workers=4,
                                              callbacks=callbacks)




###################################################################
###  保存训练信息                                                ###
###################################################################

print(model_train_history.history['acc'])
print(model_train_history.history['val_acc'])
print(model_train_history.history['loss'])
print(model_train_history.history['val_loss'])


# Save
filename = "{}_result.npz".format(model_name)
save_dict = {
    "acc": model_train_history.history['acc'],
    "val_acc": model_train_history.history['val_acc'],
    "loss": model_train_history.history['loss'],
    "val_loss":model_train_history.history['val_loss']
}
output = os.path.join("./results/", filename)
np.savez(output, **save_dict)

# Plot training & validation accuracy values
plt.plot(model_train_history.history['acc'])
plt.plot(model_train_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_acc.png'.format(model_name))
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/{}_loss.png'.format(model_name))
plt.show()


prediction_classes = model.predict(X_test)
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))



# return ax
filename = './images/{}_confusion_matrix.png'.format(model_name)
# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, filename=filename, normalize=False,
                      title='confusion matrix')
