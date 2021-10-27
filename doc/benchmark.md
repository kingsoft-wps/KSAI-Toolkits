# benchmark
本文基于KSAI-Lite推理框架给出了开源, OCR及NLP系列模型精度指标和在各平台采用不同加速方法预测耗时的benchmark。
## 模型
### 开源模型
* [MobilieNet V2](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/MobileNet_V2/MobileNet_V2.zip)

包含了浮点模型以及定点量化模型
### 自研模型
#### CV模型
* [Mobile OCR文本检测](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/CV/mobilenet_east.tflite)
* [Mobile OCR文字识别](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/CV/line_recog_MINICNN_CN.tflite)
* [边缘检测模型](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/CV/rect_detection.tflite)
* [图片分类模型](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/CV/doc_image_cf.20180907_11.2.4.tflite)
#### NLP模型
* [textcnn模型](http://sdk.ai.wpscdn.cn/KSAI/OpenSource/models/NLP/textcnn_7_dc9595852c652f6b3b3be3c0123d3624.zip)
## 平台描述
### 国产化平台：
#### 华为 Kunpeng 920：
* 处理器指令集架构：ARMV8
* 操作系统：UOS （类linux）
* 内存：8G
* 处理器：2.6G * 4 / 8核
* 算力（GFLOPS）：
* 上市时间：2019.7

#### 龙芯Loongson-3A R3(Loongson-3A3000)：
* 处理器指令集架构：MIPS64
* 操作系统：中标（类linux）
* 内存：8G
* 处理器：1500MHz * 4核
* 算力（GFLOPS）：24
* 上市时间：2017

#### 兆芯： C-QuadCore C4600
* 处理器指令集架构：X86
* 操作系统：中标（类linux）
* 内存：4G
* 处理器：2000MHz * 4核
* 算力（GFLOPS）：
* 上市时间：2016年末
* 详情链接：
#### 飞腾： Phytium FT1500a 64bit
* 处理器指令集架构：ARMV8
* 操作系统：银河（类linux）
* 内存：4G
* 处理器：1500MHz * 4核 ？？or 16核
* 算力（GFLOPS）：
* 上市时间：July 26, 2016

### X86处理器
#### 英特尔I5-9500
* 处理器指令集架构：X86-64
* 操作系统：win10
* 内存：16G
* 处理器：3000MHz * 6核
* 算力（GFLOPS）：
* 上市时间：2019年Q2

### 移动端ARM处理器
#### 高通骁龙845
* 处理器指令集架构：ARMV8
* 操作系统：ANDROID
* 内存：8G
* 处理器：2.8GHZ * 8
* 算力（GFLOPS）：
* 上市时间：2017年底
## 加速方法
### xnnpack
#### 简介
xnnpack是google推出的针对CPU实现高度优化的浮点运算，针对移动端，采用 ARM NEON优化；针对 x86-64 设备，添加对SSE2、SSE4、AVX、AVX2和 AVX512指令集的优化；而且还做了算子融合。官方给出浮点推理的速度平均可提升 2.3 倍。具体可以参见官方链接：https://discuss.tf.wiki/t/topic/465

## 性能统计
### 说明
本此性能统计，是从不同网络，不同平台，不同加速方法三个维度进行统计。

### 各网络大小统计

|        模型        | size（MB） |
| :----------------: | :--------: |
|   MobilieNet V2    |    13.3    |
| Mobile OCR文本检测 |    14.4    |
|    边缘检测模型    |    4.6     |
|    图片分类模型    |    3.1     |
|    textcnn模型     |    1.5     |

### 未使用任何加速方法各网络推理时间（ms）

|        模型        | 英特尔I5-9500（linux） | 高通845 | 华为 Kunpeng 920 | Loongson-3A3000 | 兆芯C4600 | Phytium FT1500a |
| :----------------: | :--------------------: | :-----: | :--------------: | :-------------: | :-------: | :-------------: |
|   MobilieNet V2    |         60.524         |  32.12  |     37.2886      |     444.329     |  300.129  |     119.073     |
| Mobile OCR文本检测 |        108.652         |         |     80.2075      |     880.375     |  541.008  |     248.783     |
|    边缘检测模型    |         32.641         |  28.5   |      36.94       |     359.084     |  164.66   |     124.103     |
|    图片分类模型    |        27.2178         |  28.25  |      16.982      |       204       |  134.335  |     53.453      |
|    textcnn模型     |        135.862         |   53    |     142.792      |      1452       |  635.134  |     362.688     |

### 采用xnnpack加速后各网络推理时间（ms）

|        模型        | 英特尔I5-9500（linux） | 高通845 | 华为 Kunpeng 920 | Loongson-3A3000 | 兆芯C4600 | Phytium FT1500a |
| :----------------: | :--------------------: | :-----: | :--------------: | :-------------: | :-------: | :-------------: |
|   MobilieNet V2    |        13.3443         |  17.25  |     27.8156      |        -        |  108.861  |     89.991      |
| Mobile OCR文本检测 |        30.2953         |    -    |     59.7572      |        -        |     -     |     193.562     |
|    边缘检测模型    |        16.9135         |  21.25  |      34.401      |        -        |  146.823  |     123.252     |
|    图片分类模型    |         6.5623         |  25.87  |      12.621      |        -        |  51.772   |     53.333      |
|    textcnn模型     |        22.9107         |   36    |      41.373      |        -        |  159.858  |     362.036     |

### 总体性能统计
## OCR数据集
* 图像数据(IP : http://sdk.ai.wpscdn.cn/KSAI/OCR_data/input/image.zip)
* label信息(IP : http://sdk.ai.wpscdn.cn/KSAI/OCR_data/input/golden/annotation_jsonfile.zip)