

创新思考

#### 1. 能否有代替TF-IDF的编码？

#### 2. 概率总和1如何做到？

#### 3. 如何提高KNN算法的搜索效率

### 思考题

#### 1. IDF计算公式中分母为什么需要+1？

$$
idf_j = log\frac{N}{1+\sum_{j=1}^{N} v_j},\ v_j=1\ if\ word_j\ appeared\ in\ text_n
$$



#### 2.IDF数值有什么含义，TF-IDF数值有什么含义？

IDF可能可以作为一个词是否代表特殊语意的标准，**IDF值越小**说明这个词在所有不同文档出现的次数较多，越可能**不**传达有效信息（比如说the、I等通用名词），从这个角度看，IDF数值可以代替一个词对文档含义影响程度的权重值。

#### 3.为什么KNN距离的倒数作为权重？如何使同一个测试样本的各个情感概率总和为1？

$$
assume: \ n\ is\ the\ number\ of\ train\ set,\ and\ k\ is\ the\ number\ of\ labels \newline
we\ have\  \sum_{j=1}^{k} p_{ij} = 1, \newline
then,\ \sum_{j=1}^{k}\sum_{i=1}^{n}{\frac{p_{ij}}{d_i}}=\sum_{i=1}^{n}\sum_{j=1}^{k}\frac{p_{ij}}{d_i}=\sum_{i=1}^{n}\frac{1}{d_i} \newline
so,\ we\ can\ divide\ probability\ with\ the\ result\ above,\ to\ make\ the\ sum\ is\ 1.
$$





#### 实验结果

Classification: 

![截屏2021-09-09 下午2.58.44](/Users/liuxb/Desktop/截屏2021-09-09 下午2.58.44.png)

![](/Users/liuxb/Desktop/截屏2021-09-09 下午3.03.33.png)

Regression:

<img src="/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-09 下午3.06.31.png" alt="截屏2021-09-09 下午3.06.31" style="zoom:50%;" />

![截屏2021-09-09 下午3.07.57](/Users/liuxb/Library/Application Support/typora-user-images/截屏2021-09-09 下午3.07.57.png)

