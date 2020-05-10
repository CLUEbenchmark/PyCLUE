# PyCLUE

Python toolkit for Chinese Language Understanding Evaluation benchmark.

中文语言理解测评基准的Python工具包，快速测评代表性数据集、基准（预训练）模型，并针对自己的数据选择合适的基准（预训练）模型进行快速应用。

## 关于CLUE

datasets, baselines, pre-trained models, corpus and leaderboard

[中文语言理解测评基准](https://www.cluebenchmarks.com/)，包括代表性的数据集、基准(预训练)模型、语料库、排行榜。

我们会选择一系列有一定代表性的任务对应的数据集，做为我们测试基准的数据集。这些数据集会覆盖不同的任务、数据量、任务难度。

## 安装PyCLUE

现在，可以通过pip安装PyCLUE：

```bash
pip install --upgrade PyCLUE
```

或直接git clone安装PyCLUE：

```bash
pip install git+https://www.github.com/CLUEBenchmark/PyCLUE.git
```

## 基准（预训练）模型

**已支持预训练语言模型**

1. [BERT-zh](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
2. [BERT-wwm-ext](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip)
3. [albert_xlarge_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip)
4. [albert_large_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_large_zh.zip)
5. [albert_base_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_base_zh.zip)
6. [albert_base_ext_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip)
7. [albert_small_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip)
8. [albert_tiny_zh_brightmart](https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip)
9. [roberta_zh_brightmart](https://storage.googleapis.com/chineseglue/pretrain_models/roeberta_zh_L-24_H-1024_A-16.zip)
10. [roberta_wwm_ext_zh_brightmart]('https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_ext_L-12_H-768_A-12.zip)
11. [roberta_wwm_ext_large_zh_brightmart](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip)

**待支持**

1. [XLNet_mid](https://github.com/ymcui/Chinese-PreTrained-XLNet)
2. [ERNIE_base](https://github.com/PaddlePaddle/ERNIE)

## 快速评测CLUE数据集

### 数据集介绍及下载

**注：数据集与[CLUEBenchmark](https://github.com/CLUEbenchmark/CLUE)所提供的数据集一致，仅在格式上相应修改，以适应PyCLUE项目**

#### 1. AFQMC 蚂蚁金融语义相似度

##### 数据介绍

```
数据量：训练集（34334）验证集（4316）测试集（3861）
例子：
{"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
每一条数据有三个属性，从前往后分别是 句子1，句子2，句子相似度标签。其中label标签，1 表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。
```

链接：https://pan.baidu.com/s/1It1SiMJbsrNl1dEOBoOGXg 
提取码：ksd1

##### 测评脚本

训练模型脚本位置：PyCLUE/clue/sentence_pair/afqmc/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/afqmc/train.ipynb

提交文件脚本位置：PyCLUE/clue/sentence_pair/afqmc/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/afqmc/predict.ipynb

#### 2. TNEWS' 今日头条中文新闻（短文本）分类 Short Text Classificaiton for News

##### 数据介绍

该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

```
数据量：训练集(266,000)，验证集(57,000)，测试集(57,000)
例子：
{"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。
```

链接：https://pan.baidu.com/s/1Rs9oXoloKgwI-RgNS_GTQQ 
提取码：s9go

##### 测评脚本

训练模型脚本位置：PyCLUE/clue/classification/tnews/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/classification/tnews/train.ipynb

提交文件脚本位置：PyCLUE/clue/classification/tnews/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/classification/tnews/predict.ipynb

#### 3. IFLYTEK' 长文本分类 Long Text classification

##### 数据介绍

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。

```
数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)
例子：
{"label": "110", "label_des": "社区超市", "sentence": "朴朴快送超市创立于2016年，专注于打造移动端30分钟即时配送一站式购物平台，商品品类包含水果、蔬菜、肉禽蛋奶、海鲜水产、粮油调味、酒水饮料、休闲食品、日用品、外卖等。朴朴公司希望能以全新的商业模式，更高效快捷的仓储配送模式，致力于成为更快、更好、更多、更省的在线零售平台，带给消费者更好的消费体验，同时推动中国食品安全进程，成为一家让社会尊敬的互联网公司。,朴朴一下，又好又快,1.配送时间提示更加清晰友好2.保障用户隐私的一些优化3.其他提高使用体验的调整4.修复了一些已知bug"}
每一条数据有三个属性，从前往后分别是 类别ID，类别名称，文本内容。
```

链接：https://pan.baidu.com/s/1EKtHXmgt1t038QTO9VKr3A 
提取码：u00v

##### 评测脚本

训练模型脚本位置：PyCLUE/clue/classification/iflytek/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/classification/iflytek/train.ipynb

提交文件脚本位置：PyCLUE/clue/classification/iflytek/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/classification/iflytek/predict.ipynb

#### 4. CMNLI 语言推理任务 Chinese Multi-Genre NLI

##### 数据介绍

CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，对原始MNLI数据和XNLI数据进行了中英文转化，保留原始训练集，合并XNLI中的dev和MNLI中的matched作为CMNLI的dev，合并XNLI中的test和MNLI中的mismatched作为CMNLI的test，并打乱顺序。该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。

```
数据量：train(391,782)，matched(12,426)，mismatched(13,880)
例子：
{"sentence1": "新的权利已经足够好了", "sentence2": "每个人都很喜欢最新的福利", "label": "neutral"}
每一条数据有三个属性，从前往后分别是 句子1，句子2，蕴含关系标签。其中label标签有三种：neutral，entailment，contradiction。
```

链接：https://pan.baidu.com/s/1mFT31cBs2G6e69As6H65dQ 
提取码：kigh

##### 评测脚本

训练模型脚本位置：PyCLUE/clue/sentence_pair/cmnli/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/cmnli/train.ipynb

提交文件脚本位置：PyCLUE/clue/sentence_pair/cmnli/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/cmnli/predict.ipynb

#### 5. 诊断集 CLUE_diagnostics test_set

##### 数据介绍

诊断集，用于评估不同模型在9种语言学家总结的中文语言现象上的表现。

使用在CMNLI上训练过的模型，直接预测在这个诊断集上的结果，提交格式和CMNLI一致，在排行榜详情页可以看到结果。（注：该数据集包含CMNLI的训练集与测试集）

链接：https://pan.baidu.com/s/1DYDUGO6xN_4xAT0Y4aNsiw 
提取码：u194

##### 评测脚本

训练模型脚本位置：PyCLUE/clue/sentence_pair/diagnostics/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/diagnostics/train.ipynb

提交文件脚本位置：PyCLUE/clue/sentence_pair/diagnostics/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/clue/sentence_pair/diagnostics/predict.ipynb

#### 6. 其他CLUE支持的数据集

补充中。

## 应用于自定义任务

#### 1. 多分类任务 Multi Class Classification

##### 任务说明

多分类任务，如文本分类、情感分类等，可接受单句输入和句子对输入两种形式。

##### 数据要求

数据目录下应至少包含train.txt，dev.txt和labels.txt文件，可增加test.txt文件。

保存形式参考：

单句输入（对应评测脚本中的`task_type = 'single'`）：PyCLUE/examples/classification/single_data_templates/，https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/classification/single_data_templates

句子对输入（对应评测脚本中的`task_type = 'pairs'`）：PyCLUE/examples/classification/pairs_data_templates/，https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/classification/pairs_data_templates

**注：应采用\t作为分隔符。**

##### 评测脚本

训练模型脚本位置：PyCLUE/examples/classification/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/classification/train.ipynb

预测使用脚本位置：PyCLUE/examples/classification/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/classification/predict.ipynb

#### 2. 句子对任务（孪生网络） Sentence Pair (Siamese)

##### 任务说明

句子对任务（孪生网络），如相似句子对任务等。**与多分类任务中的句子对输入模型区别：多分类任务中的句子对任务采用类似Bert的拼接形式进行输入，而该任务采用孪生网络的形式进行输入。**

##### 数据要求

数据目录下应至少包含train.txt，dev.txt和labels.txt文件，可增加test.txt文件。

保存形式参考：

输入：PyCLUE/examples/sentence_pair/data_templates/，https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/sentence_pair/data_templates

**注：应采用\t作为分隔符。**

##### 评测脚本

训练模型脚本位置：PyCLUE/examples/sentence_pair/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/sentence_pair/train.ipynb

预测使用脚本位置：PyCLUE/examples/sentence_pair/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/sentence_pair/predict.ipynb

#### 3. 文本匹配任务（孪生网络） Text Matching (Siamese)

##### 说明

文本匹配任务（孪生网络），如FAQ检索、QQ匹配检索等任务，使用孪生网络生成输入句子的embedding信息，使用[hnswlib](https://github.com/nmslib/hnswlib)检索最相近的若干句子。

##### 数据要求

数据目录下应至少包含cache.txt，train.txt，dev.txt和labels.txt文件，可增加test.txt文件。

保存形式参考：

输入：PyCLUE/examples/text_matching/data_templates/，https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/text_matching/data_templates

**注：应采用\t作为分隔符。**

##### 评测脚本

训练模型脚本位置：PyCLUE/examples/text_matching/train.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/text_matching/train.ipynb

预测使用脚本位置：PyCLUE/examples/text_matching/predict.ipynb

参考：https://github.com/CLUEBenchmark/PyCLUE/blob/master/examples/text_matching/predict.ipynb

## 训练生成文件

#### 1. 模型文件

模型文件包含10个最新的checkpoint模型文件和pb模型文件（10个checkpoint模型文件在测试集dev.txt上表现最佳的模型）。

![训练生成文件](https://i.loli.net/2020/05/10/7bZIvJakD8x1tGl.png)

#### 2. 训练过程指标

训练过程生成的指标文件（train_metrics.png），分别为accuracy，total_loss，batch_loss，precision，recall和f1指标。

<img src="https://i.loli.net/2020/05/10/gkS2GPyClDNrjuK.png" alt="train_metrics" style="zoom:200%;" />

#### 3. 验证过程指标

若存在验证文件test.txt且验证文件各行以true_label作为起始，则打印最佳模型在验证文件上的指标。

![image-20200510133813806](https://i.loli.net/2020/05/10/bpzCFT2t8GBOunk.png)

## API文档

更新中。

## 其他说明

正式地址：https://github.com/CLUEBenchmark/PyCLUE

调试地址：https://github.com/liushaoweihua/PyCLUE

## Timeline

### 更新日志

* 2019.12.05
  * [初版PyCLUE](https://github.com/chineseGLUE/PyCLUE)，用以快速评测CLUE数据集（文本分类、句子对任务）；
* 2020.05.10
  * 代码改版，合并冗余代码（测试版本：tensorflow 1.15.2），为简化API，在下游任务上暂时移除对TPU的支持；
  * 支持多版bert、albert和roberta模型，可根据指定预训练语言名自动下载并加载使用；
  * 支持文本分类、句子对、文本匹配任务；
  * 用以快速评测CLUE数据集（AFQMC/TNEWS/IFLYTEK/CMNLI），生成[CLUEBenchmark](https://www.cluebenchmarks.com/)可接受的提交文件；
  * 应用于自定义任务，快速快速生成checkpoint和tensorflow-serving支持部署的pb模型文件形式，并可加载pb模型文件进行预测；支持文件形式质检，保存误识别结果至指定目录。

### 更新计划

* 2020.05 ~ 2020.08
  * 支持其他文本分类、句子对和文本匹配任务；
  * 支持序列标注任务；
  * 支持XLNET、ERNIE、ELECTRA等；
  * 支持预训练词向量模型（Word2Vec等），支持多类下游网络；
* 2020.08 ~ 2020.10
  * 支持阅读理解任务；
  * 支持TF 2.0；
* 2020.10 ~ 2020.12
  * 对接[NLPCC 2020 LightLM高性能小模型评测项目](https://github.com/CLUEbenchmark/LightLM)，支持多类小模型；
  * 整合[CLUE已支持的Pytorch模型](https://github.com/CLUEbenchmark/CLUE/tree/master/baselines/models_pytorch)。