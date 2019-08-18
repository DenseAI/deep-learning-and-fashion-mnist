import os

from keras.datasets import mnist, fashion_mnist
#from utils.loaders import load_mnist
from CAE import Autoencoder
import matplotlib.pyplot as plt
import numpy as np

def load_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() #mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


# run params
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = './run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE =  'build' #'load' #

(x_train, y_train), (x_test, y_test) = load_mnist()

AE = Autoencoder(
    input_dim = (28,28,1)
    , encoder_conv_filters = [32,64,64, 64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,2,2,1]
    , decoder_conv_t_filters = [64,64,32,1]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [1,2,2,1]
    , z_dim = 128
)

if MODE == 'build':
    AE.save(RUN_FOLDER)
else:
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


x_train_append = []
y_train_append = []
y_train_random_append = []
num_classes = 10
for ii in range(x_train.shape[0]):
    x = x_train[ii]
    y = y_train[ii]
    #x_train_append(x)
    for jj in range(num_classes):
        x_train_append.append(x)
        y_train_append.append(y)
        y_train_random_append.append(jj)

x_train_append = np.array(x_train_append)
y_train_append = np.array(y_train_append)
y_train_random_append = np.array(y_train_random_append)

print(x_train_append.shape)
print(y_train_append.shape)
print(y_train_random_append.shape)


AE.encoder.summary()
AE.decoder.summary()

LEARNING_RATE = 0.0005
BATCH_SIZE = 256
INITIAL_EPOCH = 0
EPOCHS = 20

AE.compile(LEARNING_RATE)

model_train_history = AE.train(x_train_append, y_train_append,
                               x_train_append, y_train_random_append,
                               x_test, y_test,
                               batch_size = BATCH_SIZE,
                               epochs = EPOCHS,
                               run_folder = RUN_FOLDER,
                               initial_epoch = INITIAL_EPOCH)

# Plot training & validation accuracy values

#print(model_train_history.history)
plt.plot(model_train_history.history['model_2_acc_1'])
plt.plot(model_train_history.history['val_model_2_acc_1'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/cae_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/cae_loss.png')
plt.show()