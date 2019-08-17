# Fashion-Mnist数据集基线

在[zalandoresearch](https://github.com/zalandoresearch/fashion-mnist)官方库，列举了部分基线算法，我们挑选部分算法进行实现。

## 目录
- [1 卷积神经网络(CNN)](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#1-卷积神经网络cnn)
- [2 MobileNet](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#1-卷积神经网络cnn)

## 1 卷积神经网络(CNN)


## 2 MobileNet
@苏剑林的Fashion-Mnist例子，我修改了MobileNet，在不加载预训练权重，准确率（最高为0.9322）如下：
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_acc.png" "mobilenet_acc">
</p>
Loss如下：
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_confusion_loss.png" "mobilenet_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/baseline/mobilenet/images/mobilenet_confusion_matrix.png" "mobilenet_acc">
</p>
详细报表如下：
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
