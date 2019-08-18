# 深度学习与Fashion-Mnist数据集

通过Fashion-Mnist数据集，发现问题、寻找解决办法、并快速进行算法验证

## 目录
- [1 数据集基线](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)

## 1 数据集基线
在[zalandoresearch](https://github.com/zalandoresearch/fashion-mnist)官方库，列举了部分基线算法，我们挑选部分算法进行实现。
<p>
CNN:
</>
<p align="center">
  <img width="640" src="/baseline/cnn/images/cnn_acc.png" "cnn_acc">
</p>
<p>
MobileNet(加载预训练权重):
</>
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_acc2.png" "mobilenet_acc">
</p>
<p>
混淆矩阵:
</>
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_confusion_matrix.png" "mobilenet_acc">
</p>


## 参考

- [zalandoresearch/fashion-mnist ](https://github.com/zalandoresearch/fashion-mnist)
- [fashion mnist的一个baseline (MobileNet 95%) ](https://kexue.fm/archives/4556)
- [Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)
- [zhunzhong07/Random-Erasing ](https://github.com/zhunzhong07/Random-Erasing)