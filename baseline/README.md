# Fashion-Mnist数据集基线

在[zalandoresearch](https://github.com/zalandoresearch/fashion-mnist)官方库，列举了部分基线算法，我们挑选部分算法进行实现。

## 目录
- [1 卷积神经网络(CNN)](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#1-卷积神经网络cnn)
- [2 MobileNet](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#1-卷积神经网络cnn)
- [3 MobileNet](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#1-卷积神经网络cnn)
## 1 卷积神经网络(CNN)
在[Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)的例子，准确率（最高为0.9415）如下：
<p align="center">
  <img width="640" src="/baseline/cnn/images/cnn_acc.png" "cnn_acc">
</p>
Loss如下：
<p align="center">
  <img width="640" src="/baseline/cnn/images/cnn_loss.png" "cnn_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/baseline/cnn/images/cnn_confusion_matrix.png" "cnn_acc">
</p>
详细报表如下：

```
              precision    recall  f1-score   support

           0       0.92      0.85      0.88      1000
           1       1.00      0.99      0.99      1000
           2       0.91      0.90      0.91      1000
           3       0.94      0.93      0.94      1000
           4       0.88      0.94      0.91      1000
           5       0.99      0.99      0.99      1000
           6       0.80      0.82      0.81      1000
           7       0.97      0.98      0.98      1000
           8       0.99      0.99      0.99      1000
           9       0.98      0.97      0.97      1000

   micro avg       0.94      0.94      0.94     10000
   macro avg       0.94      0.94      0.94     10000
weighted avg       0.94      0.94      0.94     10000
```

## 2 MobileNet
[@苏剑林](http://kexue.fm/archives/4556/)的Fashion-Mnist例子，我修改了MobileNet，在不加载预训练权重，准确率（最高为0.9322，在加载预训练权重下可以最高0.9440左右）如下：
未加载预训练权重
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_acc.png" "mobilenet_acc">
</p>
加载预训练权重:
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_acc2.png" "mobilenet_acc">
</p>
Loss如下：
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_loss.png" "mobilenet_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_confusion_matrix.png" "mobilenet_acc">
</p>
详细报表如下：

```
              precision    recall  f1-score   support

           0       0.88      0.85      0.86      1000
           1       0.99      0.99      0.99      1000
           2       0.90      0.91      0.91      1000
           3       0.90      0.96      0.93      1000
           4       0.91      0.88      0.90      1000
           5       0.99      0.99      0.99      1000
           6       0.79      0.80      0.79      1000
           7       0.96      0.98      0.97      1000
           8       0.99      0.98      0.99      1000
           9       0.98      0.96      0.97      1000

    micro avg       0.93      0.93      0.93     10000
    macro avg       0.93      0.93      0.93     10000
 weighted avg       0.93      0.93      0.93     10000
```


## 3 Wide Resnet
在[Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)的例子，准确率最高的前三位分别是
0.967、0.963、0.959，分别使用了Wide Resnet，我们参考了 @zhunzhong07 的例子，使用Keras实现了Wide Resnet，但准确率达不到Pytorch版的WideResnet，准确率如下：
<p align="center">
  <img width="640" src="/baseline/wrn/images/wide_resnet_acc.png" "wide_resnet_acc">
</p>
Loss如下：
<p align="center">
  <img width="640" src="/baseline/wrn/images/wide_resnet_loss.png" "wide_resnet_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/baseline/wrn/images/wide_resnet_confusion_matrix.png" "wide_resnet_acc">
</p>
我们只跑了50轮，在[测试例子中](https://github.com/zhunzhong07/Random-Erasing/issues/9)，第152轮开始达到0.950，并在272轮时达到0.9580的最高值。



## 4 问题汇总

#### 问题1 分类间相互干扰
1.1 分类间相互干扰，特别0、2、4、6之间，它们之间的干扰如何形成？在[Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)
分析了部分预测错误的例子：
<p align="center">
  <img width="928" src="/baseline/images/predicted_false.png" "predicted_false">
</p>

#### 类别标注
每个训练和测试样本都按照以下类别进行了标注：

| 标注编号 | 描述 |
| --- | --- |
| 0 | T-shirt/top（T恤）|
| 1 | Trouser（裤子）|
| 2 | Pullover（套衫）|
| 3 | Dress（裙子）|
| 4 | Coat（外套）|
| 5 | Sandal（凉鞋）|
| 6 | Shirt（汗衫）|
| 7 | Sneaker（运动鞋）|
| 8 | Bag（包）|
| 9 | Ankle boot（踝靴）|

1.2 T恤0与汗衫6，相互干扰最大，通过肉眼观察，T恤是短袖且无扣子/无拉链，汗衫部分短袖，但有扣子或者拉链。
1.3 套衫2与汗衫6，通过肉眼观察，套衫是长袖且无扣子/无拉链，汗衫部分长袖，但有扣子或者拉链。
1.4 套衫2与外套4，通过肉眼观察，套衫是长袖且无扣子/无拉链，外套长袖，但有扣子或者拉链，而且布料很厚。
1.5 外套4与汗衫6，通过肉眼观察，外套长袖布料很厚？汗衫材料薄？
1.6 运动鞋7与踝靴9，通过肉眼观察，运动鞋与踝靴都有高帮鞋，但材料不一样，踝靴多是皮质材料。