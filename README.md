# ADA-DD
Digits Detection for Algorithm Design and Analysis

梁堃昌泰
肖静虹
范小晟

### Dependencies
- mxnet
- opencv
- numpy
- sklearn
- pillow
- others(see if there're import errors)

### Method
##### Pre-process
考虑到实际测试数据的特点，输入图像做了以下预处理后被传入数字检测：
1. 裁剪上方525像素（图标和汉字）
2. 放缩分辨率到1/2（减少像素数，加快处理）
3. 反色（使大部分像素为0）
4. RGB值以128为中心拉伸2倍并裁剪到0-255（加大颜色区别）
5. HSV值中色相H小于88的全改为黑色（去横线）
6. 采用椭圆形形态学运算核，进行2次膨胀，1次腐蚀（连接断开的字，同时加粗）

##### Detection
参考了[4]中的方法，使用了opencv中的CascadeClassifier进行数字的识别。识别模型也采用了[4]中提供的预训练模型（在7000个正样和9000个负样上训练）。  

##### Classification
考虑到多方面因素，模型采用了适用于cifar10的ResNet-8，实现参考了[2]。训练数据合并了[5][6]中的训练和测试集，而测试使用了提供的数据生成器里的样本。训练后从所有epoch中选取了测试准确率最高的模型作为使用的模型。调用代码稍作修改后与[4]中代码结合，替换其中的SVM分类器。

### How to Use
只需执行`python demo.py`按GUI提示操作即可。结果同时以图片输出到result.jpg以及以文本的bbox坐标加类别和置信度输出到result.txt。图片结果对应原图的1/2大小，文本结果也对应于1/2大小的像素位置。

### Directory structure
```
.
├─ README.md
├─ demo.py (entry script)
├─ aug (deprecated, augmentation for MNIST)
│   └── ...
├─ ddr (main scripts for detection and recognition)
│   └── ...
├─ dev (example test data)
│   └── ...
├─ LeNet (training script for LeNet(deprecated) and ResNet)
│   └── ...
└─ rcnn (deprecated, trail for rcnn detection)
    └── ...
```

### References
[1] https://github.com/apache/incubator-mxnet/tree/master/example/image-classification  
[2] https://github.com/tornadomeet/ResNet  
[3] https://github.com/apache/incubator-mxnet/tree/master/example/rcnn  
[4] https://github.com/bikz05/digit-recognition
[5] http://yann.lecun.com/exdb/mnist/  
[6] https://www.nist.gov/srd/nist-special-database-19
