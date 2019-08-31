
# -*- coding:utf-8 -*-


from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Helper libraries
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

if K.image_data_format() == 'channels_first':
    x_train_with_channels = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val_with_channels = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    x_test_with_channels = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_with_channels = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val_with_channels = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    x_test_with_channels = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print("train feature shape = ", x_train_with_channels.shape)
print("validation feature shape = ", x_val_with_channels.shape)
print("test feature shape = ", x_test_with_channels.shape)


x_train_with_channels = x_train_with_channels.astype("float32") / 255.0
x_val_with_channels = x_val_with_channels.astype("float32") / 255.0
x_test_with_channels = x_test_with_channels.astype("float32") / 255.0

y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)




###################################################################
###  创建模型                                                    ###
###################################################################

kernels = [3,3]
model = create_base_cnn_model_with_kernels(input_shape, kernels=kernels, optimizer="adamax")
model.summary()



model_name = "base_cnn_error_learning"
loss_value = 'val_acc'
checkpoint_path = './weights/{}_weight.ckpt'.format(model_name)
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    # Early stopping definition
    EarlyStopping(monitor=loss_value, patience=10, verbose=1),
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
epochs = 50
data_augmentation = False
pixel_level = True
Training = False
Fine_tuning = True

if load:
    model.load_weights(checkpoint_path)

if Training:
    if not data_augmentation:
        model_train_history = model.fit(x_train_with_channels, y_train_categorical,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_val_with_channels, y_val_categorical),
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
            preprocessing_function=get_random_eraser(probability = 0.33))

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train_with_channels)

        # Fit the model on the batches generated by datagen.flow().
        model_train_history = model.fit_generator(datagen.flow(x_train_with_channels, y_train_categorical),
                                                  steps_per_epoch=x_train_with_channels.shape[0] // batch_size,
                                                  validation_data=(x_val_with_channels, y_val_categorical),
                                                  epochs=epochs,
                                                  verbose=1,
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


epochs = 1
if Fine_tuning:
    #for epoch in range(epochs):
    for epoch in range(epochs):
        prediction_classes = model.predict(x_test_with_channels)
        y_pred = np.argmax(prediction_classes, axis=1)
        y_pred_categorical = keras.utils.to_categorical(y_pred, num_classes)
        print(classification_report(y_test, y_pred))


        counter = 0
        error = 0

        pred_counter = 0
        pred_error = 0
        print("prediction_classes: ", prediction_classes.shape)

        x_train_preds = []
        y_train_preds = []


        for ii in range(len(prediction_classes)):
            if(prediction_classes[ii][y_pred[ii]] > 0.99):
                x_train_preds.append(x_test_with_channels[ii])
                y_train_preds.append(y_pred[ii])

        x_train_preds = np.array(x_train_preds)
        y_train_preds = np.array(y_train_preds)
        y_train_preds = keras.utils.to_categorical(y_train_preds, num_classes)

        for ii in range(len(prediction_classes)):
            if(prediction_classes[ii][y_pred[ii]] <= 0.99):
                if (y_pred[ii] == y_test[ii]):
                    counter = counter + 1
                else:
                    error = error + 1

                sim_all = np.array(prediction_classes[ii])
                sorted_dela_errors = np.argsort(sim_all)
                most_important_errors = sorted_dela_errors[-3:]
                #print(most_important_errors)

                print(y_pred[ii], " ", y_test[ii], " ", most_important_errors, sorted_dela_errors, prediction_classes[ii][sorted_dela_errors])

                max_preds = {}
                for kk in range(len(most_important_errors)):
                    if prediction_classes[ii][most_important_errors[kk]] < 0.1:
                        continue
                    x_test_append = []
                    y_test_append = []

                    #x_test_append_2 = []
                    #y_test_append_2 = []
                    for jj in range(batch_size):
                        x_test_append.append(x_test_with_channels[ii])
                        y_test_append.append(most_important_errors[kk])

                        #x_test_append_2.append(x_test_with_channels[ii])
                        #y_test_append_2.append(sorted_dela_errors[1])
                        #break

                    x_test_append = np.array(x_test_append)
                    y_test_append = np.array(y_test_append)
                    #x_test_append_2 = np.array(x_test_append_2)
                    #y_test_append_2 = np.array(y_test_append_2)
                    y_test_append_categorical = keras.utils.to_categorical(y_test_append, num_classes)
                    #y_test_append_2_categorical = keras.utils.to_categorical(y_test_append_2, num_classes)

                    model_1 = keras.models.clone_model(model)
                    model_1.set_weights(model.get_weights())
                    model_1.compile(loss=keras.losses.categorical_crossentropy,
                                    optimizer="sgd",
                                    metrics=['accuracy'])

                    # model_2 = keras.models.clone_model(model)
                    # model_2.set_weights(model.get_weights())
                    # model_2.compile(loss=keras.losses.categorical_crossentropy,
                    #                 optimizer="sgd",
                    #                 metrics=['accuracy'])



                    model_train_history_1 = model_1.fit(np.vstack([x_test_append, x_train_with_channels]), np.vstack([y_test_append_categorical, y_train_categorical]),
                                                        batch_size=batch_size,
                                                        epochs=5,
                                                        verbose=2,
                                                        validation_data=(x_train_preds, y_train_preds))

                    # model_train_history_2 = model_2.fit(np.vstack([x_test_append_2, x_train_with_channels]), np.vstack([y_test_append_2_categorical, y_train_categorical]),
                    #                                     batch_size=batch_size,
                    #                                     epochs=5,
                    #                                     verbose=2,
                    #                                     validation_data=(x_val_with_channels, y_val_categorical))

                    max_preds[most_important_errors[kk]] = max(model_train_history_1.history['val_acc'])

                pred = y_pred[ii]
                # if max(model_train_history_1.history['val_acc']) > max(model_train_history_2.history['val_acc']):
                #     pred = most_important_errors[0]
                # else:
                #     pred = most_important_errors[1]

                max_acc = -1000000
                for (d, x) in max_preds.items():
                    if x >= max_acc:
                        pred = d
                        max_acc = x


                if(pred == y_test[ii]):
                    pred_counter = pred_counter + 1
                else:
                    pred_error = pred_error + 1

                print(y_pred[ii], " ", y_test[ii], " ", pred)

    print("counter: ", counter)
    print("error: ", error)
    print("pred_counter: ", pred_counter)
    print("pred_error: ", pred_error)




###################################################################
###  模型分析                                                    ###
###################################################################
# prediction_classes = model.predict(x_test_with_channels)
# y_val_pred = np.argmax(prediction_classes, axis=1)
#
#
# prediction_threshold = 0.99
# val_pred_under_results = {}
# val_pred_upper_results = {}
#
# val_pred_under_counters = {}
# val_pred_upper_counters = {}
#
# val_pred_counters = {}
#
# for ii in range(len(x_test_with_channels)):
#     if y_val_pred[ii] != y_test[ii]:
#         if prediction_classes[ii][y_val_pred[ii]] > prediction_threshold:
#             # if y_val_pred[ii] not in val_pred_upper_results.keys():
#             #     val_preds = {}
#             #     val_preds[y_val[ii]] = 1
#             #     val_pred_upper_results[y_val_pred[ii]] = val_preds
#             # else:
#             #     val_preds = val_pred_upper_results[y_val_pred[ii]]
#             #     if y_val[ii] in val_preds.keys():
#             #         val_pred = val_preds[y_val[ii]]
#             #         val_pred = val_pred + 1
#             #         val_preds[y_val[ii]] = val_pred
#             #     else:
#             #         val_preds[y_val[ii]] = 1
#             #         val_pred_upper_results[y_val_pred[ii]] = val_preds
#
#             if "wrong" in val_pred_upper_counters.keys():
#                 val_pred_upper_counters["wrong"] = val_pred_upper_counters["wrong"] + 1
#             else:
#                 val_pred_upper_counters["wrong"] = 1
#
#         else:
#             # if y_val_pred[ii] not in val_pred_under_results.keys():
#             #     val_preds = {}
#             #     val_preds[y_val[ii]] = 1
#             #     val_pred_under_results[y_val_pred[ii]] = val_preds
#             # else:
#             #     val_preds = val_pred_under_results[y_val_pred[ii]]
#             #     if y_val[ii] in val_preds.keys():
#             #         val_pred = val_preds[y_val[ii]]
#             #         val_pred = val_pred + 1
#             #         val_preds[y_val[ii]] = val_pred
#             #     else:
#             #         val_preds[y_val[ii]] = 1
#             #         val_pred_under_results[y_val_pred[ii]] = val_preds
#
#             if "wrong" in val_pred_under_counters.keys():
#                 val_pred_under_counters["wrong"] = val_pred_under_counters["wrong"] + 1
#             else:
#                 val_pred_under_counters["wrong"] = 1
#
#
#         # 记录每个分类的错误分类个数
#         if y_test[ii] not in val_pred_counters.keys():
#             val_counters = {}
#             val_counters["wrong"] = 1
#             val_pred_counters[y_test[ii]] = val_counters
#         else:
#             val_counters = val_pred_counters[y_test[ii]]
#             if "wrong" in val_counters.keys():
#                 val_counter = val_counters["wrong"]
#                 val_counter = val_counter + 1
#                 val_counters["wrong"] = val_counter
#             else:
#                 val_counters["wrong"] = 1
#
#     else:
#         # 记录每个分类的正确分类个数
#         if y_test[ii] not in val_pred_counters.keys():
#             val_counters = {}
#             val_counters["right"] = 1
#             val_pred_counters[y_test[ii]] = val_counters
#         else:
#             val_counters = val_pred_counters[y_test[ii]]
#             if "right" in val_counters.keys():
#                 val_counter = val_counters["right"]
#                 val_counter = val_counter + 1
#                 val_counters["right"] = val_counter
#             else:
#                 val_counters["right"] = 1
#
#
#         if prediction_classes[ii][y_val_pred[ii]] > prediction_threshold:
#             if "right" in val_pred_upper_counters.keys():
#                 val_pred_upper_counters["right"] = val_pred_upper_counters["right"] + 1
#             else:
#                 val_pred_upper_counters["right"] = 1
#         else:
#             if "right" in val_pred_under_counters.keys():
#                 val_pred_under_counters["right"] = val_pred_under_counters["right"] + 1
#             else:
#                 val_pred_under_counters["right"] = 1
#
#     # if prediction_classes[ii][y_val_pred[ii]] > prediction_threshold:
#     #     if y_val_pred[ii] not in val_pred_upper_results.keys():
#     #         val_preds = {}
#     #         val_preds[y_val[ii]] = 1
#     #         val_pred_upper_results[y_val_pred[ii]] = val_preds
#     #     else:
#     #         val_preds = val_pred_upper_results[y_val_pred[ii]]
#     #         if y_val[ii] in val_preds.keys():
#     #             val_pred = val_preds[y_val[ii]]
#     #             val_pred = val_pred + 1
#     #             val_preds[y_val[ii]] = val_pred
#     #         else:
#     #             val_preds[y_val[ii]] = 1
#     #             val_pred_upper_results[y_val_pred[ii]] = val_preds
#     #
#     # else:
#     if y_val_pred[ii] not in val_pred_under_results.keys():
#         val_preds = {}
#         val_preds[y_test[ii]] = 1
#         val_pred_under_results[y_val_pred[ii]] = val_preds
#     else:
#         val_preds = val_pred_under_results[y_val_pred[ii]]
#         if y_test[ii] in val_preds.keys():
#             val_pred = val_preds[y_test[ii]]
#             val_pred = val_pred + 1
#             val_preds[y_test[ii]] = val_pred
#         else:
#             val_preds[y_test[ii]] = 1
#             val_pred_under_results[y_val_pred[ii]] = val_preds
#
#     # # 记录每个分类的错误分类个数
#     # if y_val[ii] not in val_pred_counters.keys():
#     #     val_counters = {}
#     #     val_counters["wrong"] = 1
#     #     val_pred_counters[y_val[ii]] = val_counters
#     # else:
#     #     val_counters = val_pred_counters[y_val[ii]]
#     #     if "wrong" in val_counters.keys():
#     #         val_counter = val_counters["wrong"]
#     #         val_counter = val_counter + 1
#     #         val_counters["wrong"] = val_counter
#     #     else:
#     #         val_counters["wrong"] = 1
#
#
# print(val_pred_upper_results)
# print(val_pred_under_results)
#
# print("val_pred_upper_counters: ", val_pred_upper_counters)
# print("val_pred_under_counters: ", val_pred_under_counters)
#
# print(val_pred_counters)
#
# filename = './images/{}_confusion_matrix.png'.format(model_name)
# plot_confusion_matrix(y_test, y_val_pred, classes=classes, filename=filename, normalize=False,
#                       title='confusion matrix')
# print(classification_report(y_test, y_val_pred))
#
#
#
# prediction_classes = model.predict(x_test_with_channels)
# y_pred = np.argmax(prediction_classes, axis=1)
#
#
# # return ax
# filename = './images/{}_confusion_matrix.png'.format(model_name)
# # Plot confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=classes, filename=filename, normalize=False,
#                       title='confusion matrix')
#
# print(classification_report(y_test, y_pred))
#
#
#
#
# pred_right_counter = 0
# pred_change_right_counter = 0
#
# counter = 0
# for ii in range(len(x_test_with_channels)):
#     if prediction_classes[ii][y_pred[ii]] <= prediction_threshold:
#         counter += 1
#         if y_pred[ii] in val_pred_under_results.keys():
#
#             # 随机选择 大于错误率 的预测进行修改
#             rnd = random.random()
#             #print(rnd)
#             #print(val_pred_under_counters["wrong"] / (val_pred_under_counters["wrong"] + val_pred_under_counters["right"]))
#             #if rnd > (val_pred_under_counters["right"] / (val_pred_under_counters["wrong"] + val_pred_under_counters["right"])):
#
#             #对其他分类预测错误的数量
#             val_pred_under = val_pred_under_results[y_pred[ii]]
#
#             wrong_sum = 0.0
#             for k, v in val_pred_under.items():
#                 wrong_sum = wrong_sum + val_pred_under[k]
#
#             dict = sorted(val_pred_under.items(), key=lambda d: d[1], reverse=True)
#             #print(wrong_sum, " ", dict)
#
#             select_rnd = random.random()
#
#             select_index = y_pred
#
#             prob = 0.0
#             jj = 0
#             for v in dict:
#
#                 if prob <  select_rnd < prob + (float(v[1]) / wrong_sum):
#                     select_index = int(v[0])
#                     #break
#                 #print(select_rnd, " ", prob, " ", prob + (float(v[1]) / wrong_sum))
#                 prob = prob + (float(v[1]) / wrong_sum)
#
#             print(y_pred[ii], " ", y_test[ii], " ", select_index)
#
#             if(y_pred[ii] == y_test[ii]):
#                 pred_right_counter += 1
#
#             if (select_index == y_test[ii]):
#                 pred_change_right_counter += 1
# print("counter: ", counter)
# print("pred_right_counter ", pred_right_counter)
# print("pred_change_right_counter ", pred_change_right_counter)
#
#










