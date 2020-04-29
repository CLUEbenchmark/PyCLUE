# PyCLUE 个人开发版本

## 正式版本

CLUEBenchmark-PyCLUE: https://github.com/CLUEBenchmark/PyCLUE

之前写的时候处于学习阶段，代码冗余较为严重。

当前版本对原始的PyCLUE进行了重构，目前暂时可支持：

1. 一键下载、安装、解压、转换权重和重命名部分预训练语言模型（bert和albert，后续支持xlnet，ernie，electra等）；

2. 暂时写了三个demo：文本多分类，相似句子对任务（孪生网络），文本匹配（孪生网络+hnsw），后续支持其他nlp任务；

3. example中，run_train.py文件训练模型，支持异步（先在训练集上训练若干个epoch，再在最后保留的5个checkpoint上从开发集上选择最优参数）和同步（同时训练和验证参数），并将最优模型序列化为pb文件，训练过程中打印日志、绘制指标训练结果等；run_predict.py文件加载训练好的pb文件进行预测；run_quality_inspection.py指定文件进行模型验证，输出错误预测的文本结果；

4. 目前暂时支持tensorflow 1.x，后续尝试支持tensorflow 2.x和pytorch；

5. 过段时间写一下使用手册，目前仍需进行api的调整。
