## AI第一次实验—KNN的实现

#### 18308133 刘显彬 计科二班

#### 

### 一、实验原理

#### 1.KNN原理

KNN是一种邻近预测算法，也属于监督学习的一种（需要有标记的数据），通过收集训练数据作为基准数据，当需要评估某个（些）测试数据时，就逐一计算其与基准数据的相似度，取出最为相似的一组（k个）基准数据，通过某些权重加成（如测试数据和这组基准数据的相似度作为权重）以及依照某种规则（比如投票：选出现标记最多的一个）选出其中一个标记作为该测试数据的预测结果。其中：

带有标记的**数据**可以用其特征值+标记值组成，而测试数据只有特征值，特征可有多个。

**相似度**可以用数据间的（各种）距离来衡量：如果数据的特征值可以量化的话，我们就可以利用量化后的排列有序的特征值（特征向量）来表征该数据的属性， 此时我们就可以对两个特征向量求出各种距离：曼哈顿距离、欧式距离、余弦距离等：其中各种距离公式如下：
$$
假设: x_i, x_j是两个量化后的特征向量，特征个数（维度）为n，则有：\newline
L_p(x_i,x_j)=\{\sum_{k=1}^{n}\left | x_{ik}-x_{jk} \right|^{p}\}^{\frac{1}{p}},\ 其中p=1为曼哈顿距离，2为欧式距离 \newline
余弦距离公式：cos(x_i, x_j)=\frac{x_i*x_j}{\left|x_i\right|\left|x_j\right|}
$$

#### 2.利用KNN模型的流程

##### 2.1 由上面的分析，我们可以总结出运用**KNN模型**需要的实现：

1. KNN需要**存储**（大量的）已标记的数据用于训练模型
2. 数据处理：需要对数据的特征进行**量化**便于分析数据间的距离度量，特别地，在这个实验中选用的是**tf-idf**编码的方式进行数据处理
3. 距离函数：选择合适的距离函数可以加强特征的利用（个人认为：比如欧式距离弱化了各个特征的区别，在空间上是一个球面，即各个方向的类似），实验中同时完成了上述提到的三种距离
4. 预测函数：将测试数据传入KNN模型，利用距离函数计算出测试数据和所有训练数据的距离，取出与测试数据最相似的k个数据，根据k个标记中最多的一个作为测试数据的标记。

##### **2.2 其中，KNN模型需要调整两个超参数：k和距离函数类型，后面会分析到不同参数值对模型的影响，在正式将我们的模型用于预测之前，我们需要根据某些流程选出可能的、最合适的模型参数：**

1. 将已经标记好的数据集划出一部分用于考察参数选择的效果，这部分数据称为验证集(Validation Set)，其余部分（**不相交**）就是我们的训练数据集(train set)；**对所有数据进行预处理（量化）**
2. 设定好KNN模型的超参数（k和距离函数），将训练集传入模型存储；将验证集数据传入模型并调用预测函数，预测这些数据的**标记**，并与真实标记进行对比，计算出预测的正确率，正确率即可作为这个超参数配置下的模型的评估分数；
3. 重复（2），找出表现最好的一组超参数配置，这组配置即可以用于测试数据的预测

### 二、伪代码


$$
1.TF-IDF:
$$


```python
def TF-IDF(sentences):
  words_doc = get_different_word(sentences)			# 数出句子中所有不同的单词
  n = len(words_doc)	# 词表长度
  k = len(sentences)
  sentence_one_hot = int[k][n]
  for sentence_i in sentences:
    sentence_one_hot[i] = [0]*n
    # 转换成one-hot编码，其中index求取word_j在词表的位置
    for word_j in sentence:	
      sentence_one_hot[i][index(words_doc, word_j)] += 1
  TF = sentence_one_hot / sum(sentence_one_hot)		# 归一化
  # 完成TF构建
  
  # IDF：
  words_count = [0]*n
  for sentence in sentences:
    for word in sentence:
    	if word not exist since this cycle:			# 如果该单词在此句子没出现过，则计数+1
        words_count[index(word)] += 1
  IDF = log(n / (1+ words_count))						  # 转化成idf
  return TF*IDF
```


$$
2.Predict\_with\_k:
$$

```python
def predict(X, k, distance):					# 需要：测试数据，距离函数，k值
  dist = distance(training_data, X)		# 根据距离公式计算测试数据X和训练数据的距离
  K_index = min_K(dist, k)						# 获取前K个最近的训练数据
  K_labels= training_data[K_index]		# 取得他们的label
  return countMax(K_labels)						# 找出最多的一个并返回
                      
```


$$
3.improve\_predict\_for\_regression:\\
$$

```python
def predict_regression(X, K, distance):
  dist = distance(training_data, X)		# 根据距离公式计算测试数据X和训练数据的距离
  K_index = min_K(dist, k)						# 获取前K个最近的训练数据
  K_labels= training_data[K_index]		# 取得他们的label
  K_dist = dist[K_index]							# 取得他们的距离
  factor = sum(1/K_dist)							# 放缩因子为：距离倒数和
  return sum(K_label, weight=1/ K_dist)*factor	# 乘以权重相加
```

$$
4.整体流程
$$

```python
model=KNN()
test_data, train_data, val_data = TF-IDF(sentences)	# 转化为TF-IDF矩阵
model.train(train_data)
for param in params_possible:												# 找最佳参数
  predict_result=predict(val_data, param)						# 预测val data 结果
  performance[param] = evalution(predict_result)		# 评估这组参数
 	
findBest(performance) # 找到最佳参数
test_result = model.predict(test_data, bestParam)		# 预测test data的结果并存储
save(test_result)
  
```



### 三、关键代码展示

#### 1.TF-IDF

1.1将句子根据词汇表编码成TF向量，其中词汇表存储了每一个单词对应的one-hot编码非零的下标，tf矩阵加上一个偏移是因为后面两个实验进行归一化的时候会出现分母为0的情况，所以加上一个小的偏移保证非零，本个实验中偏移值是0。

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午10.01.55.png" alt="截屏2021-09-15 下午10.01.55" style="zoom:50%;" />

1.2 IDF：统计词汇出现的文章数

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午10.07.36.png" alt="截屏2021-09-15 下午10.07.36" style="zoom:50%;" />

#### 2.distances

2.1 p=1，街区距离（曼哈顿距离）直接对应位置相减绝对值求和；

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.05.49.png" alt="截屏2021-09-15 下午5.05.49" style="zoom: 50%;" />

2.2 p=2，欧式距离*向量版，向量版原理在稍后给出

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.08.55.png" alt="截屏2021-09-15 下午5.08.55" style="zoom:50%;" />

2.3 余弦距离，非矩阵运算形式

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.12.53.png" alt="截屏2021-09-15 下午5.12.53" style="zoom:50%;" />

​	完全向量化：

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.14.00.png" alt="截屏2021-09-15 下午5.14.00" style="zoom: 50%;" />

#### 3.classification

3.1 预测：得到距离最小的k个样本，并利用bincount来对不同的label计数，利用argmax取出数目最多的label

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.18.39.png" alt="截屏2021-09-15 下午5.18.39" style="zoom:50%;" />

3.2 训练：调整不同的超参数，利用validation set对不同参数配置进行准确率评估，选择出最优的一组参数对test Set进行预测

#### <img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.23.07.png" alt="截屏2021-09-15 下午5.23.07" style="zoom:50%;" />

#### 4.regression

4.1 修改predict：取到最近的K个样本的**距离**，令他们的倒数作为权重，乘以他们的label值再相加，得到测试数据的估测向量（因此在这里的label是一个N维向量，N代表了label的种类，值代表了label的概率），并除以K个距离的倒数和，将该估测向量归一化成概率。

（思考题中展示了为什么如此操作可以得到一个和为1的估测向量）。

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午7.08.26.png" alt="截屏2021-09-15 下午7.08.26" style="zoom:50%;" />

### 四、结果展示

#### 1.TF-IDF

##### 1.1  result of sample: (path:/result/18308133_liuxianbin_TFIDF_sample.csv)

##### <img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午4.59.46.png" alt="截屏2021-09-15 下午4.59.46" style="zoom:33%;" />

删除多余的0，得到：

##### <img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午9.17.35.png" alt="截屏2021-09-15 下午9.17.35" style="zoom:33%;" />

与助教师兄/师姐给的参考一致：

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午5.02.14.png" alt="截屏2021-09-15 下午5.02.14" style="zoom: 50%;" />

##### 1.2 result of test set:(path:/result/18308133_liuxianbin_TFIDF.csv)

semeval.txt转化成tf-idf结果如下：

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午9.18.53.png" alt="截屏2021-09-15 下午9.18.53" style="zoom: 33%;" />



#### 2.classification

##### 2.1 result of test set:

调整超参数时的输出，输出数据是对应不同k和距离函数下的准确率，其中losstype为1，2，3分别表示曼哈顿距离、欧式距离、余弦距离，可以看出对该实验validation Set而言，k=7，距离函数取余弦距离时，KNN模型表现最好，准确率可以达到0.463（稍后将分析为什么不同k和距离函数会导致模型的不一样）

##### <img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午7.32.38.png" alt="截屏2021-09-15 下午7.32.38" style="zoom:33%;" />

下面是test_set的预测结果：(path: /result/18308133_liuxianbin_classification.csv)

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午7.49.00.png" alt="截屏2021-09-15 下午7.49.00" style="zoom: 33%;" />



#### 3.regression

##### 3.1 result of validation set:

这里由于label不再是单一的离散值，而是各种label的估计概率，因此这里改用相关系数来代替准确率作为模型评价，这里的最佳相关系数为0.406（对六个label取平均），这里导出了最佳的validation Set的预测结果**（path: /result/validation_predict.csv)**，放入评估相关系数的EXCEL表格的结果见第二个图

<img src="/Users/liuxb/Desktop/截屏2021-09-15 下午8.00.36.png" alt="截屏2021-09-15 下午8.00.36" style="zoom:33%;" />

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午8.51.15.png" alt="截屏2021-09-15 下午8.51.15" style="zoom:33%;" />

##### 3.2 result of test set:

##### 结果部分截图如下：<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-15 下午9.01.04.png" alt="截屏2021-09-15 下午9.01.04" style="zoom: 33%;" />



### 五、改进和结果分析

##### 5.1 计算各种距离时，用上向量化的形式计算距离矩阵，提高效率

######  如欧式距离向量化推导：

$$
|X-Y|^2 = \sum(x_i-y_i)^2=\sum{x_i}^2+\sum{y_i}^2-2\sum{x_i*y_i}=|X|^2+|Y|^2-X*Y^T
$$

##### 5.2 为什么不同的K会出现不同的表现

个人理解是：过小的K会被training的样本的噪声干扰，会被一些非正常的偏移值引导，增加K可以增加抗干扰能力；过大的K会导致在投票选取最终label的时，所有测试数据都出现一样的结果，因为K过大，会使得距离较远的样本也参与了无效的”投票“

##### 5.3 为什么不同距离函数会出现不同表现

原因其实在距离函数分析那里已经有所说明，不同距离函数青睐的特征不同，从而根据不同数据的特征向量的特点选择不同的距离函数。

### 六、思考题

#### 1. IDF计算公式中分母为什么需要+1？

$$
idf_j = log\frac{N}{1+\sum_{j=1}^{N} v_j},\ v_j=1\ if\ word_j\ appeared\ in\ text_n
$$

但分母中的词频可能会为0，如果没有这个1的偏移的话，会出现除以一个0的计算错误；事实上，在处理TF-IDF矩阵的时候，我也用到了这个偏移的办法消除0值。

#### 2.IDF数值有什么含义，TF-IDF数值有什么含义？

IDF可能可以作为一个词是否代表特殊语意的标准，**IDF值越小**说明这个词在所有不同文档出现的次数较多，越可能**不**传达有效信息（比如说the、I等通用名词），从这个角度看，IDF数值可以代替一个词对文档含义影响程度的权重值。

#### 3.为什么KNN距离的倒数作为权重？如何使同一个测试样本的各个情感概率总和为1？

$$
设：k是label的种数，n是测试样本个数\newline
we\ have\  \sum_{j=1}^{k} p_{ij} = 1, \newline
then,\ \sum_{j=1}^{k}\sum_{i=1}^{n}{\frac{p_{ij}}{d_i}}=\sum_{i=1}^{n}\sum_{j=1}^{k}\frac{p_{ij}}{d_i}=\sum_{i=1}^{n}\frac{1}{d_i} \newline
所以我们可以在除以该放缩因子，使得每个样本的预测概率估计和为1
$$

