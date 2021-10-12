### Introduction

KSAI-ToolKits

KSAI-Toolkits is a toolkit from AI group of Kingsoft Office Corporation. The inference engine is KSAI-Lite (derived from TensorFlow Lite). The toolkit supports OCR function on multiple OS platforms, including Windows, Linux x86_64, and Linux ARM, at the time of initial release. With time, we plan to add support for more functionalities in the future.

### Quick Start

#### 1. Environment setup

##### Requirement on different OS platforms
###### OS
* Linux: ubuntu 16.04
* Windows: windows 10
###### CMake
The CMake version: 3.16+
###### Compiler:
* Linux: GCC 5.4+
* Windows: Visual Studio 2019
#### 2. Clone this repository
```
git clone https://github.com/kingsoft-wps/KSAI-Toolkits.git
``` 

#### 3. Setup 3rd party libraries
* Download libraries from server (IP: https://sdk.ai.wpscdn.cn/KSAI/KSAI-Toolkits/3rdparty.zip)
* Decompress the package, and put the 3rdparty folder in the root of the repository.

The downloaded package mainly contains:
* opencv lib and the header files 
* KSAI-Lite lib and the header files
* Clipper source code
#### 4. Download models and test data
* Download models and test data from server(IP: https://sdk.ai.wpscdn.cn/KSAI/KSAI-Toolkits/data.zip)
* Decompress the package, and put the data folder in the root of the repository

#### 5. Build the project

the build scripts are in tools folder

* build script on windows is build_windows.bat
```
build_windows.bat
``` 
* build script on linux x86 is build_linux.sh
```
./build_linux.sh
``` 
* build script on arm is build_aarch64.sh
```
./build_aarch64.sh
``` 

#### 6.  Run the test program

Once the compilation is sucessfully finished, look for the executable file KSAI_Toolkits_demo in /output/$platform$/bin/ folder. You can execute it with the following command line paremeters:
```
./output/linux/bin/KSAI_Toolkits_demo data/OCR/model/db_171_varied.tflite data/OCR/model/cls_best.tflite data/OCR/model/mv3_345000.tflite data/OCR/dictionary/char_6599.txt data/OCR/input/test.jpg
```
The output of the program should be:
```
finish the 0 process!! and cost 5.87865s
boxes size:55
绝密★考试结束前
2017年11月浙江省普通高校招生 选考科日考试
历史试题
2017.11
选择题部分
一、选择题（本大题共 30小题，每小题 2分，共 60分。每小题列出的四个备选
项中只有一个是符合题目要求的，不选、多选、错选均不得分）
1.古代有学者论及中国早期国家的的政治制度，谓∶
“周之子孙，苟不狂感者，莫不为天下
之显诸侯。”这反映了
A神权与王权相结合
B 最高执政集团权力的高度集中
D 政治权力的分配采用分封制和宗法制
C 血缘关系亲统不再作为权力分配依据
2.在中国思想文化第个枝繁叶茂期，有思想家针对“百家异说”局面，认为人的认识应
当力避“私（偏爱）其所积，唯恐闻其恶也;倚其所私，以观异术，唯恐闻其美也”
其观点旨在表达
A倡导独立思考精神
.不同学派应相互竞争
C 吸收各家思想精华
D 理论认识应格物致知
3.水利是农业的命脉。右图所示人物主持的水利工程闻名于世，历苍黄风雨，
惠泽中华民族两子余载，联称世界水利工程的典范。这 ‘水利工程是
A灵渠
B.都江堰 
C 郑国渠
D 白渠
李冰石像
4.有学者认为，古代丝绸之路的意义不仅使中国的丝绸远销罗马为中心的地中海世界，更
大的贡献还在于沟通了东西方经济文化交流。下列项中反映东西方交流的有
①中国造纸术经丝路传到欧洲
②美洲马铃薯、玉米传到欧洲
④中国瓷器经丝路远销欧洲
③中国印刷沿海上丝路传到日本
A①③
C ②③
D.②④
B. ①④
“自此以来，公聊大夫士土吏彬彬多文学
5.汉武帝时期，设立中央官学，培养《五经》博士，
之士矣”，中央官学的建立
A 推动了儒家思想正统地位的确立
B 结束了大富豪子嗣垄断官位的局面
C 有利于学生思想创新和个性发展
D 促进了百家争鸣局面的进 步发展
6.韩愈谈到人生“喜怒窘穷，忧悲愉铁，怨恨思慕，酣醉无聊不平，有动于心”时，提到
些文人往往通过 种书体挥洒性情，
某书法大家作品被其誉为“变动犹鬼神，不可端
倪”，共称 代法书。韩愈所赞誉的书体是
A小繁
B.楷体
D.草书
C 行书
7.中国古代皇帝制度建立后，围绕集权与分权，官僚体制不断调整和变化。下列与“分宰
第1页
```
### How to build your own program using KSAI_xxxxx
Take a look at ./test/test_ocr.cpp. 

What you need to do is:
* Include ks_ocr.h file in your project.
* Import libKSAI-Toolkits.lib/libKSAI-Toolkits.dll/libKSAI-Toolkits.so and related 3rd party libs into your project reference.
* Do whatever you want with the KSAIxxx classes.
