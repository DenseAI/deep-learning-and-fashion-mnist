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

#Keras Mobilenet，进行部分修改，主要是不加载预训练的权重
from mobilenet import MobileNet

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
base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')

output = Dropout(0.25)(base_model.output)
predict = Dense(10, activation='softmax')(output)

model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


X_train = X_train.reshape((-1,28,28))
X_train = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_train))])/255.

X_test = X_test.reshape((-1,28,28))
X_test = np.array([misc.imresize(x, (height,width)).astype(float) for x in tqdm(iter(X_test))])/255.


#图片反转
def random_reverse(x):
	if np.random.random() > 0.5:
		return x[:,::-1]
	else:
		return x

#生成数据
def data_generator(X,Y,batch_size=100):
	while True:
		idxs = np.random.permutation(len(X))
		X = X[idxs]
		Y = Y[idxs]
		p,q = [],[]
		for i in range(len(X)):
			p.append(random_reverse(X[i]))
			q.append(Y[i])
			if len(p) == batch_size:
				yield np.array(p),np.array(q)
				p,q = [],[]
		if p:
			yield np.array(p),np.array(q)
			p,q = [],[]

#进行训练
model_train_history = model.fit_generator(data_generator(X_train,y_train),
										  steps_per_epoch=int(len(X_train)/100),
										  epochs=50,
										  validation_data=data_generator(X_test,y_test),
										  validation_steps=100)


# Plot training & validation accuracy values
plt.plot(model_train_history.history['acc'])
plt.plot(model_train_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/mobilenet_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(model_train_history.history['loss'])
plt.plot(model_train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('./images/mobilenet_loss.png')
plt.show()

print(model_train_history)

prediction_classes = model.predict(X_test)
prediction_classes = np.argmax(prediction_classes, axis=1)
print(classification_report(y_test, prediction_classes))



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	#classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout();
	plt.savefig('./images/mobilenet_confusion_matrix.png')
	#return ax

# Plot confusion matrix
plot_confusion_matrix(y_test, prediction_classes, classes=classes, normalize=False,
                      title='confusion matrix')



