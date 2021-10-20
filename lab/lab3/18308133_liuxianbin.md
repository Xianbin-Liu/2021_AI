# 感知机与逻辑回归

### 18308133 刘显彬

## 1.实验原理

### 1.1 感知机

### 1.2 逻辑回归



## 2.伪代码

## 3.关键代码分析

### 3.1 PLA 训练模块

开始遍历所有训练数据，逐个进行预测，$iter$记录迭代次数（这里理解为修正的误分类点的次数），$error\_num$是记录对所有数据的一次完整遍历中的错误数，

```python
iter = 0
while (iter < iters):
  error_num = 0       # record the misclassification times over each look of the whole dataset
  # training on all dataset
  for epoch in range(len(trainset)):
    data, label = trainset[epoch], labels[epoch] # get one single data
    if iter >= iters:
      break
      # if misclassification-->update iteration times, error times and W
      if self.predSingle(data) != 2*label-1:       # convert{0,1}->{-1,1} using map: y=2x-1
        iter += 1
        error_num += 1
        # dW = - y_i*x_i
        self.W += (lrate*(2*label-1)*data[:self.dims]).reshape(-1,1)
        # while error_rate in trainset smaller than given threshold, BREAK
        if error_num/len(trainset) <= error_rate : break 
```



## 4.实验结果与分析



## 5.思考题

### 5.1 随机梯度下降与批量梯度下降各自的优缺点？



### 5.2 不同的学习率$\eta$对模型收敛有何影响？从收敛速度和是否收敛两方面来回答。



### 5.3 使用梯度的模长是否为零作为梯度下降的收敛终止条件是否合适，为什么？一般如何判断模型收敛？



## 6.遇到的问题