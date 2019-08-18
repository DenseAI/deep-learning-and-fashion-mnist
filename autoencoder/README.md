# 根因分析-生成模式

通过Fashion-Mnist数据集的算法基线分析，初步发现了一些问题，这里进行根因分析，寻找解决办法、并快速进行算法验证

## 目录
- [1 自编码器AutoEncoder](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [2 带分类条件的自编码器AC-AutoEncoder](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [3 量子化自编码器VQ-VAE](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [4 基于Triplet loss的编码器](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)

## 1 自编码器AutoEncoder
首先，我们采用简单的AutoEncoder：

<p align="center">
  <img width="700" src="/autoencoder/AE/images/autoencoder.jpg" "Auto Encoder">
</p>

分类采用Cosine Similarity的计算方式，预测时与训练集比较，取相似度最高的样本的分类：
```
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)
```

混淆矩阵如下：
<p align="center">
  <img width="640" src="/autoencoder/AE/images/ae_confusion_matrix.png" "mobilenet_acc">
</p>

从上面的结果看，简单Auto Encoder，无法满足分类要求。



## 2 带分类的自编码器AC-AutoEncoder
单纯的AutoEncoder，没有包含分类信息，所以准确率比较低，所以我们在AutoEncoder的基础上，包含分类信息
```
	### THE ENCODER
	encoder_input = Input(shape=self.input_dim, name='encoder_input')

	x = encoder_input

	for i in range(self.n_layers_encoder):
		conv_layer = Conv2D(
			filters=self.encoder_conv_filters[i],
			kernel_size=self.encoder_conv_kernel_size[i],
			strides=self.encoder_conv_strides[i],
			padding='same',
			name='encoder_conv_' + str(i)
		)

		x = conv_layer(x)
		x = LeakyReLU()(x)

		if self.use_batch_norm:
			x = BatchNormalization()(x)

		if self.use_dropout:
			x = Dropout(rate=0.25)(x)

	shape_before_flattening = K.int_shape(x)[1:]

	x = Flatten()(x)
	encoder_output = Dense(self.z_dim, name='encoder_output')(x)

	#包含分类信息
	label = Input(shape=(1,), dtype='int32')
	label_embedding = Flatten()(Embedding(self.num_classes, self.z_dim)(label))
	encoder_output = multiply([encoder_output, label_embedding])

	self.encoder = Model([encoder_input, label], encoder_output)
```
准确率如下：
<p align="center">
  <img width="640" src="/autoencoder/CAE/images/cae_acc.png" "cae_acc">
</p>
Loss如下：
<p align="center">
  <img width="640" src="/autoencoder/CAE/images/cae_loss.png" "cae_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/autoencoder/CAE/images/cae_confusion_matrix.png" "cae_acc">
</p>


## 3 量子化自编码器VQ-VAE

## 4 基于Triplet loss的编码器

## 参考

- [zalandoresearch/fashion-mnist ](https://github.com/zalandoresearch/fashion-mnist)
- [fashion mnist的一个baseline (MobileNet 95%) ](https://kexue.fm/archives/4556)
- [Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)
- [zhunzhong07/Random-Erasing ](https://github.com/zhunzhong07/Random-Erasing)