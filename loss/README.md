# 根因分析-Loss函数

通过分析Fashion-Mnist数据集算法基线中的Loss函数，发现问题根因、寻找解决办法、并快速进行算法验证

## 目录
- [1 SoftMax](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [2 SphereFace](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)
- [3 ArcFace](https://github.com/DenseAI/deep-learning-and-fashion-mnist#1-数据集基线)

## 1 SoftMax
在基线算法中卷积神经网络，采用了SoftMax函数，

准确率如下：
<p align="center">
  <img width="640" src="/loss/softmax/images/softmax_acc.png" "softmax_acc">
</p>
混淆矩阵如下：
<p align="center">
  <img width="640" src="/loss/softmax/images/softmax_confusion_matrix.png" "softmax_acc">
</p>

卷积神经网络作为基线算法，准确率最高为94.15%。

## 2 SphereFace
[1]是CVPR2017的文章，用改进的softmax做人脸识别，改进点是提出了angular softmax loss（A-softmax loss）用来改进原来的softmax loss。
M=2、4、6 的准确率如下：
<p align="center">
  <img width="640" src="/loss/arcface/images/sphereface_val_acc.png" "a-softmax_acc">
</p>

M=2 混淆矩阵如下：
<p align="center">
  <img width="640" src="/loss/arcface/images/sphere_confusion_matrix_2.png" "a-softmax_acc">
</p>

SphereFace在M=2时，准确率最高为95.32%，比卷积神经网络基线版本，提升与1.17%。

## 3 ArcFace


## 参考

- [4uiiurz1/keras-arcface ](https://github.com/4uiiurz1/keras-arcface)
- [auroua/InsightFace_TF ](https://github.com/auroua/InsightFace_TF)
- [YunYang1994/SphereFace](https://github.com/YunYang1994/SphereFace)



## 论文
- [1] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) 