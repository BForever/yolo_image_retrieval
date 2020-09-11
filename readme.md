# 图像检索流程
## 数据库建立
首先调用`detect.py`调用yolo获得数据库和待检索图片中的物体信息和深度特征
分别对应生成的文件`object_t_list.csv`，`object_e_list.csv`，`train.npz`，`eval.npz`

### 参数
- `--data`指定数据库图片文件夹/单个图片
- `--det`指定yolo打印结果（与检索无关）
- `--weights`指定yolov3.weights文件路径，yolov3的权重文件未包含在仓库中，可以使用`wget https://pjreddie.com/media/files/yolov3.weights`命令下载

其他神经网络具体参数可以在`detect.py`中查看

如果指定了文件夹，则会生成训练集信息文件，即`object_t_list.csv`和`train.npz`

如果指定了单张图片，则会生成待检索图片信息文件，即`object_e_list.csv`和`eval.npz`


## 训练和检索
调用`retrival.py`进行训练和检索，目前代码是先训练后检索一次性完成。

### 参数
- `--data`指定数据库图片文件夹
- `--query`指定待检索图片
- `--features_t`训练集特征文件train.npz路径
- `--objects_t`训练集物体信息文件object_t_list.csv路径
- `--features_e`待检索特征文件eval.npz路径
- `--objects_e`待检索物体信息文件object_e_list.csv路径

- `--num`指定要检索待检索图片中的第几个物体，从0开始，默认为0，即第一个物体
- `--pca`指定pca压缩到包含原特征的多少比例特征量，默认为0.9，将特征信息量压缩到原来的0.9
- `--dis`指定检索使用的向量间距离计算方式，有cos,l2 两种，分别是余弦距离和欧氏距离，默认为余弦距离
- `--pooling`指定如何使用物体的多个cell深度特征，有max，single和average，分别为最大池化，取最中心cell和平均池化，默认为最大池化

# 原仓库readme
## YOLO_v3_tutorial_from_scratch
Accompanying code for Paperspace tutorial series ["How to Implement YOLO v3 Object Detector from Scratch"](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

Here's what a typical output of the detector will look like ;)

![Detection Example](https://i.imgur.com/m2jwnen.png)

## About the training Code

This code is only mean't as a companion to the tutorial series and won't be updated. If you want to have a look at the ever updating YOLO v3 code, go to my other repo at https://github.com/ayooshkathuria/pytorch-yolo-v3

Also, the other repo offers a lot of customisation options, which are not present in this repo for making tutorial easier to follow. (Don't wanna confuse the shit out readers, do we?)

About when is the training code coming? I have my undergraduate thesis this May, and will be busy. So, you might have to wait for a till the second part of May. 

Cheers

