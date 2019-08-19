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

详细报表如下：
```
              precision    recall  f1-score   support

           0       0.81      0.85      0.83       100
           1       0.98      0.98      0.98        90
           2       0.80      0.81      0.80        99
           3       0.92      0.85      0.89        96
           4       0.83      0.82      0.83       102
           5       0.97      0.93      0.95       112
           6       0.69      0.71      0.70       100
           7       0.92      0.96      0.94        98
           8       0.99      0.96      0.97        99
           9       0.95      0.98      0.97       104

   micro avg       0.89      0.89      0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```

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

详细报表如下：
```
              precision    recall  f1-score   support

           0       0.83      0.89      0.86       108
           1       0.97      1.00      0.99       103
           2       0.87      0.87      0.87       103
           3       0.92      0.86      0.89       100
           4       0.84      0.84      0.84       101
           5       0.99      1.00      1.00       100
           6       0.71      0.67      0.69        82
           7       0.99      0.99      0.99       109
           8       1.00      0.99      1.00       106
           9       0.99      0.99      0.99        88

   micro avg       0.92      0.92      0.92      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.92      0.91      1000

```
在Auto-Encoder的基础上，加入分类信息后，能够提升约2%~3%，但0、2、6分类差的现象并没有多大改善。

## 3 量子化自编码器VQ-VAE

## 4 基于Triplet loss的编码器

## 参考

- [The official code repository for examples in the O'Reilly book 'Generative Deep Learning' ](https://github.com/davidADSP/GDL_code)
- [SpikeKing/triplet-loss-mnist ](https://github.com/SpikeKing/triplet-loss-mnist)
- [HenningBuhl/VQ-VAE_Keras_Implementation](https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation)
- [zhunzhong07/Random-Erasing ](https://github.com/zhunzhong07/Random-Erasing)

## 论文
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu