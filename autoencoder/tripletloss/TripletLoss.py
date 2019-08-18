# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, K, merge, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.optimizers import RMSprop
import os
import random
import warnings

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

# from bases.trainer_base import TrainerBase
# from root_dir import ROOT_DIR
# from utils.np_utils import prp_2_oh_array
# from utils.utils import mkdir_if_not_exist


from sklearn.metrics import confusion_matrix

from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K


from mobilenet import MobileNet


import os
import shutil

import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 存储项目所在的绝对路径

def prp_2_oh_array(arr):
    """
    概率矩阵转换为OH矩阵
    arr = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.6]])
    :param arr: 概率矩阵
    :return: OH矩阵
    """
    arr_size = arr.shape[1]  # 类别数
    arr_max = np.argmax(arr, axis=1)  # 最大值位置
    oh_arr = np.eye(arr_size)[arr_max]  # OH矩阵
    return oh_arr

def mkdir_if_not_exist(dir, is_delete=False):
    """
    创建文件夹
    :param dirs: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print(u'[INFO] 文件夹 "%s" 存在, 删除文件夹.' % dir)

        if not os.path.exists(dir):
            os.makedirs(dir)
            print(u'[INFO] 文件夹 "%s" 不存在, 创建文件夹.' % dir)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model, data, config, class_mode=None):
        self.model = model  # 模型
        self.data = data  # 数据
        self.config = config  # 配置
        self.class_mode = class_mode

    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError


class ModelBase(object):
    """
    模型基类
    """

    def __init__(self, config=None):
        self.config = config  # 配置
        self.model = None  # 模型
        self.class_model = None  # 模型

    def save(self, checkpoint_path):
        """
        存储checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        self.model.save_weights(checkpoint_path)
        print("[INFO] Model saved")

    def load(self, checkpoint_path):
        """
        加载checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("[INFO] Model loaded")

    def build_model(self):
        """
        构建模型
        """
        raise NotImplementedError



class DataLoaderBase(object):
    """
    数据加载的基类
    """

    def __init__(self, config):
        self.config = config  # 设置配置信息

    def get_train_data(self):
        """
        获取训练数据
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        获取测试数据
        """
        raise NotImplementedError

from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical


class TripletDL(DataLoaderBase):
    def __init__(self, config=None):
        super(TripletDL, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data() #mnist.load_data()

        # 展平数据，for RNN，感知机
        # self.X_train = self.X_train.reshape((-1, 28 * 28))
        # self.X_test = self.X_test.reshape((-1, 28 * 28))

        # X_train_filter = []
        # y_train_filter = []
        # for ii in range(self.X_train.shape[0]):
        #     if self.y_train[ii] in [0,1,2,3,4,6]:
        # 	    X_train_filter.append(self.X_train[ii])
        # 	    y_train_filter.append(self.y_train[ii])
        #
        # self.X_train = np.array(X_train_filter)
        # self.y_train = np.array(y_train_filter)
        #
        # # print(X_train.shape)
        # # print(y_train.shape)
        #
        # X_test_filter = []
        # y_test_filter = []
        # for ii in range(self.X_test.shape[0]):
        # 	if self.y_test[ii] in [0,1,2,3,4,6]:
        # 		X_test_filter.append(self.X_test[ii])
        # 		y_test_filter.append(self.y_test[ii])
        #
        # self.X_test = np.array(X_test_filter)
        # self.y_test = np.array(y_test_filter)


        # 图片数据，for CNN
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        print("[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape)))
        print("[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape)))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

class TripletModel(ModelBase):
    """
    TripletLoss模型
    """

    MARGIN = 1.0  # 超参

    def __init__(self, config=None):
        super(TripletModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model, self.class_model = self.triplet_loss_model()  # 使用Triplet Loss训练Model

    def triplet_loss_model(self):
        anc_input = Input(shape=(28, 28, 1), name='anc_input')  # anchor
        pos_input = Input(shape=(28, 28, 1), name='pos_input')  # positive
        neg_input = Input(shape=(28, 28, 1), name='neg_input')  # negative

        shared_model = self.base_model()  # 共享模型

        std_out = shared_model(anc_input)
        pos_out = shared_model(pos_input)
        neg_out = shared_model(neg_input)

        print("[INFO] model - 锚shape: %s" % str(std_out.get_shape()))
        print("[INFO] model - 正shape: %s" % str(pos_out.get_shape()))
        print("[INFO] model - 负shape: %s" % str(neg_out.get_shape()))

        output = Concatenate()([std_out, pos_out, neg_out])  # 连接

        output1 = Dense(256, activation="relu")(std_out)
        output1 = Dropout(0.5)(output1)
        output1 = Dense(10, activation="softmax")(output1)

        output2 = Dense(256, activation="relu")(pos_out)
        output2 = Dropout(0.5)(output2)
        output2 = Dense(10, activation="softmax")(output2)

        output3 = Dense(256, activation="relu")(neg_out)
        output3 = Dropout(0.5)(output3)
        output3 = Dense(10, activation="softmax")(output3)

        model = Model(inputs=[anc_input, pos_input, neg_input], outputs=[output,output1,output2,output3])

        # plot_model(model, to_file=os.path.join(self.config.img_dir, "triplet_loss_model.png"),
        #            show_shapes=True)  # 绘制模型图
        model.compile(loss=[self.triplet_loss,"categorical_crossentropy","categorical_crossentropy","categorical_crossentropy"], optimizer=Adam(), metrics=["accuracy"])

        class_model = Model(inputs=anc_input, outputs=output1)
        # Compile the model
        #class_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        return model, class_model

    @staticmethod
    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """

        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

        loss = K.maximum(basic_loss, 0.0)

        print("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
        return loss

    def base_model(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        """
        k = 4
        ins_input = Input(shape=(28, 28, 1))
        x = Conv2D(32 * k, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(ins_input)
        x = Conv2D(32 * k, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64 * k, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(x)
        x = Conv2D(64 * k, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        out = Dense(128, activation='relu')(x)

        base_model = Model(ins_input, out)
        #plot_model(model, to_file=os.path.join(self.config.img_dir, "base_model.png"), show_shapes=True)  # 绘制模型图

        # height = 56
        # width = 56
        # input_image = Input(shape=(height, width))
        # input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 3), 3, 3))(input_image)
        # base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')


        return base_model






class TripletTrainer(TrainerBase):
    def __init__(self, model, data, config=None, class_model=None):
        super(TripletTrainer, self).__init__(model, data, config, class_model)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        #self.init_callbacks()

    def init_callbacks(self):
        # train_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "train")
        # mkdir_if_not_exist(train_dir)
        # self.callbacks.append(
        #     TensorBoard(
        #         log_dir=train_dir,
        #         write_images=True,
        #         write_graph=True,
        #     )
        # )
        print("")

        # self.callbacks.append(FPRMetric())
        # self.callbacks.append(FPRMetricDetail())




    def train(self):

        from tqdm import tqdm
        from scipy import misc

        height = 56
        width = 56
        x_train = self.data[0][0]
        y_train = np.argmax(self.data[0][1], axis=1)
        x_test = self.data[1][0]
        y_test = np.argmax(self.data[1][1], axis=1)

        # x_train = x_train.reshape((-1, 28, 28))
        # x_train = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(x_train))]) #/ 255.
        #
        # x_test = x_test.reshape((-1, 28, 28))
        # x_test = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(x_test))]) #/ 255.


        y_train_raw = self.data[0][1]
        y_test_raw = self.data[1][1]

        clz_size = len(np.unique(y_train))
        print("[INFO] trainer - 类别数: %s" % clz_size)
        digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        tr_pairs, tr_labels = self.create_pairs(x_train, digit_indices, clz_size, y_train_raw)

        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        te_pairs, te_labels = self.create_pairs(x_test, digit_indices, clz_size, y_test_raw)

        anc_ins = tr_pairs[:, 0]
        pos_ins = tr_pairs[:, 1]
        neg_ins = tr_pairs[:, 2]
        y_train_anc_ins = tr_labels[:, 0]
        y_train_pos_ins = tr_labels[:, 1]
        y_train_neg_ins = tr_labels[:, 2]



        print(tr_pairs.shape)
        print(anc_ins.shape)
        print(y_train_anc_ins.shape)

        X = {
            'anc_input': anc_ins,
            'pos_input': pos_ins,
            'neg_input': neg_ins
        }

        anc_ins_te = te_pairs[:, 0]
        pos_ins_te = te_pairs[:, 1]
        neg_ins_te = te_pairs[:, 2]
        y_test_ins = te_labels

        y_test_anc_ins = te_labels[:, 0]
        y_test_pos_ins = te_labels[:, 1]
        y_test_neg_ins = te_labels[:, 2]


        X_te = {
            'anc_input': anc_ins_te,
            'pos_input': pos_ins_te,
            'neg_input': neg_ins_te
        }

        self.model.summary()
        self.class_mode.summary()

        self.model.fit(
            X, [np.ones(len(anc_ins)), y_train_anc_ins, y_train_pos_ins, y_train_neg_ins],
            batch_size=32,
            epochs=10,
            validation_data=[X_te, [np.ones(len(anc_ins_te)), y_test_anc_ins, y_test_pos_ins, y_test_neg_ins]],
            verbose=1,
            callbacks=self.callbacks)

        # model_train_history = self.model.fit_generator(self.data_generator(x_train, y_train_raw, digit_indices, clz_size),
        #                                           steps_per_epoch=int(len(x_train) / 100), epochs=10,
        #                                           validation_data=self.data_generator(x_test, y_test_raw, digit_indices, clz_size), validation_steps=100)


        self.model.save(os.path.join("./", #self.config.cp_dir,
                                    "triplet_loss_model.h5"))  # 存储模型

        y_pred = self.model.predict(X_te)  # 验证模型
        y_pred_class = y_pred[1]
        y_pred = y_pred[0]


        print(y_pred_class.shape)

        Y_pred_classes = np.argmax(y_pred_class, axis=1)
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(y_test_anc_ins, axis=1)
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
        print(confusion_mtx)



        self.show_acc_facets(y_pred, int(y_pred.shape[0] / clz_size), clz_size)

        # if(self.class_mode != None):
        #     self.class_mode.load_weights(os.path.join("./", #self.config.cp_dir,
        #                             "triplet_loss_model.h5"), by_name=True)
        #
        #
        #     self.class_mode.fit(
        #         x_train, y_train_raw,
        #         batch_size=32,
        #         epochs=2,
        #         validation_data=[x_test, y_test_raw],
        #         verbose=1,
        #         callbacks=self.callbacks)

    @staticmethod
    def show_acc_facets(y_pred, n, clz_size=10):
        """
        展示模型的准确率
        :param y_pred: 测试结果数据组
        :param n: 数据长度
        :param clz_size: 类别数
        :return: 打印数据
        """
        print("[INFO] trainer - n_clz: %s" % n)

        avg_acc = 0.0
        for i in range(clz_size):
            print("[INFO] trainer - clz %s" % i)
            final = y_pred[n * i:n * (i + 1), :]
            anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:]

            pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            basic_loss = pos_dist - neg_dist
            r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
            print("[INFO] trainer - distance - min: %s, max: %s, avg: %s" % (
                np.min(basic_loss), np.max(basic_loss), np.average(basic_loss)))
            print("[INFO] acc: %s" % (float(r_count) / float(n)))
            print("")
            avg_acc = avg_acc + (float(r_count) / float(n))
        print("[INFO]avg acc: %s" % (float(avg_acc) / float(clz_size)))

    @staticmethod
    def create_pairs(x, digit_indices, num_classes, y):
        """
        创建正例和负例的Pairs
        :param x: 数据
        :param digit_indices: 不同类别的索引列表
        :param num_classes: 类别
        :return: Triplet Loss 的 Feed 数据
        """

        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(num_classes)])  # 最小类别数
        for d in range(num_classes):
            for i in range(n):
                index_a = i
                index_b = i
                while(True):
                    index_b = random.randrange(1, len(digit_indices[d]))
                    if(index_a != index_b):
                        break

                z1, z2 = digit_indices[d][index_a], digit_indices[d][index_b]
                inc = random.randrange(1, num_classes)

                dn = (d + inc) % num_classes
                # if( d in [6] and random.randrange(0, 1) > 0.5 ):
                #     inc = random.randrange(1, 5)
                #     dn = (d + inc) % num_classes
                #else:
                #    dn = (d + inc) % num_classes

                index_c = random.randrange(1, len(digit_indices[dn]))

                #print(index_a, " ", index_b, " ", index_c, " ", dn)

                z3 = digit_indices[dn][index_c]
                pairs += [[x[z1], x[z2], x[z3]]]
                labels += [[y[z1],y[z2],y[z3]]]
        return np.array(pairs), np.array(labels)

    def data_generator(self, X, Y, digit_indices, num_classes,  batch_size=100):
        # while True:
        #     idxs = np.random.permutation(len(X))
        #     X = X[idxs]
        #     Y = Y[idxs]
        #     p, q = [], []
        #     for i in range(len(X)):
        #         p.append(random_reverse(X[i]))
        #         q.append(Y[i])
        #         if len(p) == batch_size:
        #             yield np.array(p), np.array(q)
        #             p, q = [], []
        #     if p:
        #         yield np.array(p), np.array(q)
        #         p, q = [], []

        while True:
            idxs = np.random.permutation(len(X))
            X = X[idxs]
            Y = Y[idxs]
            pairs = []
            labels = []

            n = min([len(digit_indices[d]) for d in range(num_classes)])  # 最小类别数
            for ii in range(len(X)):
                d = random.randrange(0, num_classes)
                index_a = random.randrange(0, len(digit_indices[d]))
                index_b = index_b = random.randrange(0, len(digit_indices[d]))
                # while (True):
                #     index_b = random.randrange(0, len(digit_indices[d]))
                #     if (index_a != index_b):
                #         break


                inc = random.randrange(1, num_classes)

                dn = (d + inc) % num_classes
                index_c = random.randrange(1, len(digit_indices[dn]))

                print(index_a, " ", index_b, " ", index_c, " ", dn)

                z1, z2 = digit_indices[d][index_a], digit_indices[d][index_b]
                z3 = digit_indices[dn][index_c]
                pairs += [X[z1], X[z2], X[z3]]
                labels += [Y[z1], Y[z2], Y[z3]]
                if len(pairs) == batch_size:
                    yield np.array(pairs), np.array(labels)
                    pairs, labels = [], []

            if pairs:
                yield np.array(pairs), np.array(labels)
                pairs, labels = [], []




class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print(" — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall))


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print(" — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s))


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] 解析配置...')

    parser = None
    config = None

    # try:
    #     args, parser = get_train_args()
    #     config = process_config(args.config)
    # except Exception as e:
    #     print '[Exception] 配置无效, %s' % e
    #     if parser:
    #         parser.print_help()
    #     print '[Exception] 参考: python main_train.py -c configs/triplet_config.json'
    #     exit(0)
    # config = process_config('configs/triplet_config.json')

    print('[INFO] 加载数据...')
    dl = TripletDL()

    print('[INFO] 构造网络...')
    model = TripletModel()

    print('[INFO] 训练网络...')
    trainer = TripletTrainer(
        model=model.model,
        data=[dl.get_train_data(), dl.get_test_data()],
        class_model=model.class_model
        )
    trainer.train()
    print('[INFO] 训练完成...')


if __name__ == '__main__':
    main_train()