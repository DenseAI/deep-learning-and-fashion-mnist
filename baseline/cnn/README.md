# Fashion-Mnist数据集卷积神经网络(CNN)基线

Fashion-Mnist数据集[zalandoresearch](https://github.com/zalandoresearch/fashion-mnist)的官方库，列举了部分基线算法，目前最够的准确率为0.967。
我们从最简单的CNN开始，一步步分析、一步步改进，希望能够达到或超于准确率最高的算法

## 目录
- [1 卷积神经网络(CNN)](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline/cnn#1-卷积神经网络cnn)
- [2 优化算法](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#2-mobilenet)
- [3 Wide Resnet](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#3-wide-resnet)
- [4 问题分析](https://github.com/DenseAI/deep-learning-and-fashion-mnist/tree/master/baseline#4-问题汇总)


## 1 卷积神经网络(CNN)
在[Fashion MNIST 94% Accuracy using CNN Keras](https://www.kaggle.com/albertbrucelee/fashion-mnist-94-accuracy-using-cnn-keras)的例子，准确率（最高为0.9415）如下：
<p align="center">
  <img width="500" src="/baseline/cnn/images/base_cnn_acc.png" "cnn_acc">
</p>
Loss如下：
<p align="center">
  <img width="500" src="/baseline/cnn/images/base_cnn_loss.png" "cnn_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="500" src="/baseline/cnn/images/base_cnn_confusion_matrix.png" "cnn_acc">
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
## 2 优化算法
### 2.1 随机梯度下降SGD
<p align="center">
```
x \leftarrow  x - \eta \bigtriangledown f(x)
```
</p>

