from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import wide_residual_network as wrn

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

# n_to_show = 10000
# example_idx = np.random.choice(range(len(x_train)), n_to_show)
# x_train = x_train[example_idx]
# y_train = y_train[example_idx]

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




model = wrn.create_wide_residual_network_with_label(input_shape, depth=16, nb_classes=10, k = 4, dropoutRate=0.25)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()




checkpoint_path = './weights/wrn_label_weight.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback =  ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 save_weights_only=True,
                                 period=1) #  save weights every 1 epochs

batch_size = 128
epochs = 5

load = True

if load:
	model.load_weights("./weights/wrn_label_weight.ckpt")

model_train_history = model.fit([x_train_append, y_train_random_append], y_train_append_categorical,
								batch_size=batch_size,
								epochs=epochs,
								verbose=1,
								validation_data=([x_test_with_channels, y_test_rnd], y_test_categorical),
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
plt.savefig('./images/wide_resnet_label_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/wide_resnet_label_loss.png')
plt.show()


# prediction_classes = model.predict(x_test_with_channels)
# prediction_classes = np.argmax(prediction_classes, axis=1)
# print(classification_report(y_test, prediction_classes))
#
#
# def plot_confusion_matrix(y_true, y_pred, classes,
# 						  normalize=False,
# 						  title=None,
# 						  cmap=plt.cm.Blues):
# 	"""
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
# 	if not title:
# 		if normalize:
# 			title = 'Normalized confusion matrix'
# 		else:
# 			title = 'Confusion matrix, without normalization'
#
# 	# Compute confusion matrix
# 	cm = confusion_matrix(y_true, y_pred)
# 	# Only use the labels that appear in the data
# 	# classes = classes[unique_labels(y_true, y_pred)]
# 	if normalize:
# 		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 		print("Normalized confusion matrix")
# 	else:
# 		print('Confusion matrix, without normalization')
#
# 	print(cm)
#
# 	fig, ax = plt.subplots()
# 	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
# 	ax.figure.colorbar(im, ax=ax)
# 	# We want to show all ticks...
# 	ax.set(xticks=np.arange(cm.shape[1]),
# 		   yticks=np.arange(cm.shape[0]),
# 		   # ... and label them with the respective list entries
# 		   xticklabels=classes, yticklabels=classes,
# 		   title=title,
# 		   ylabel='True label',
# 		   xlabel='Predicted label')
#
# 	# Rotate the tick labels and set their alignment.
# 	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# 			 rotation_mode="anchor")
#
# 	# Loop over data dimensions and create text annotations.
# 	fmt = '.2f' if normalize else 'd'
# 	thresh = cm.max() / 2.
# 	for i in range(cm.shape[0]):
# 		for j in range(cm.shape[1]):
# 			ax.text(j, i, format(cm[i, j], fmt),
# 					ha="center", va="center",
# 					color="white" if cm[i, j] > thresh else "black")
# 	fig.tight_layout();
# 	plt.savefig('./images/wide_resnet_label_confusion_matrix.png')
#
#
# # return ax
#
# # Plot confusion matrix
# plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,
# 					  title='confusion matrix')