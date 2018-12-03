# TextClassification

短文本分类任务

字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

#### 数据集

公司短文本分类数据集，共有14个分类。

#### 预处理

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。



#####  CNN模型

![images/cnn_architecture](https://github.com/gaussic/text-classification-cnn-rnn/raw/master/images/cnn_architecture.png )

运行 `python run_cnn.py train`，可以开始训练。



##### 测试

 运行 `python run_cnn.py test` 在测试集上进行测试。



##### 预测

为方便预测，repo 中 `predict.py` 提供了 CNN 模型的预测方法。



##### 训练过程保存模型

为了方便前端或者其他模块进行调用，我们使用Flask将其封装成API接口。

将模型保存，会生成4个文件，当然在训练的过程中除了checkpoint，其他三个文件会有多个。

```shell
checkpoint
model.data-00000-of-00001  # data文件是保存数据的(权重)
model.index  # 文件是一个不可修改的键值表
model.meta   # 文件是保存图(包括图，操作等)
```

在这个repo中，这类文件存放在```checkpoints/textcnn```中。



##### 固化训练好的模型

在训练完成后选择效果最好的模型，进行压缩，或者将graph和权重放在一起以便生产使用。

在命令行运行：```python frozen.py```

此时，模型```checkpoints/textcnn```文件夹中会多出```frozen_model.pb```文件。



其中```convert_variables_to_constants()```函数的作用是：

1. 将变量替换成常量固化起来。
2. 将前向传播不需要的节点node去掉
3. 所以`output_node_names`参数只要输入你的网络的输出，就会生成一个最小的序列化的二进制pb文件



#####  Web API

使用```flask```搭建一个微型```web Api```。

````python
app.run(host='0.0.0.0', port=5000)
````

将应用开启在5000端口，并进行广播，使局域网用户可以访问。

