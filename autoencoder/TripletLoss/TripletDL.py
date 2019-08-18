from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical


from base import DataLoaderBase, ModelBase, TrainerBase

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


