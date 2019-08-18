from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K


from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, K, merge, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.optimizers import RMSprop

import os
import shutil

import numpy as np
import os


from base import DataLoaderBase, ModelBase, TrainerBase

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

        class_model = Model(inputs=anc_input, outputs=std_out)
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


