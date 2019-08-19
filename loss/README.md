# 根因分析-Loss函数

通过分析Fashion-Mnist数据集算法基线中的Loss函数，发现问题根因、寻找解决办法、并快速进行算法验证

## 目录
- [1 SoftMax](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [2 ArcFace](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [3 根因分析-Loss函数](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)

## 1 SoftMax
在基线算法中卷积神经网络，采用了SoftMax函数，

Loss如下：
<p align="center">
  <img width="640" src="/loss/softmax/images/softmax_acc.png" "softmax_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/loss/softmax/images/softmax_confusion_matrix.png" "softmax_acc">
</p>



## 2 根因分析-生成模式

#### 2.1 自编码器AutoEncoder
#### 2.2 带分类条件的自编码器AC-AutoEncoder
#### 2.3 量子化自编码器VQ-VAE
#### 2.4 基于Triplet loss的编码器

## 3 根因分析-Loss函数

#### 3.1 SoftMax
#### 3.2 Center-Loss 
#### 3.3 L-SoftMax
#### 3.3 Norm-Loss
#### 3.4 Coco-Loss
#### 3.5 Arc-Loss

## 参考

- [zalandoresearch/fashion-mnist ](https://github.com/zalandoresearch/fashion-mnist)
- [fashion mnist的一个baseline (MobileNet 95%) ](https://kexue.fm/archives/4556)
- [Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)
- [zhunzhong07/Random-Erasing ](https://github.com/zhunzhong07/Random-Erasing)