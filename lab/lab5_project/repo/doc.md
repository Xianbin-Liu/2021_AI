神经网络主要函数：

```python
class NerualNet:
    def __init__(self, paramsFile=None, featureDim=None, outputDim=1, HiddenDims=[], actiF=0, weight_scale=1e-2) -> None:
        '''
        @paramsFile:    参数文件，如果指定了该参数文件的路径，将从该文件重建参数，而不是随机初始化；直接指出文件名即可，不需要添加后缀（如需添加后缀，请保证后缀为.npz）
        @featureDim:    输入的特征维数
        @outputDim:     输出的特征维数，如二分类或者回归预测的时候，都为1
        @HiddenDims:    中间层，请保证中间层是一个列表，列表的元素依次为每一层中间层的输入维数
        @weight_scale:  初始化特征权重，一般默认是1e-2即可，用于保证用np.random初始化的参数不要过大
        '''

    @property
    def param(self):
        '''
        @return:    返回参数字典，如：param['W1']取出W1的参数矩阵
        '''
        param = {}
        for i in range(self.layers):
            param["W"+str(i+1)] = self.W[i]
            param["b"+str(i+1)] = self.b[i]
        return param

    def eval(self):
        # 使模型关闭训练状态，请确保在预测测试集时，将训练状态关闭
        self.trainMode = False

    def trainning(self):
        # 打开训练状态，请确保在训练前，打开训练状态
        self.trainMode = True
		
    # ！！！不需要显式调用！！！
    def loadParam(self, file):
        # 加载参数文件，file为路径，默认文件结构为

    def saveParam(self, file):
        # 保存参数到指定文件路径，默认后缀为.npz
        np.savez(file, **self.param)
        
		
    # ！！ 不需要显示调用！！
    def Loss(self, X, Labels=None, epsilonmax=0.95, epsilonmin=0.05, lossfunction="RMSELoss"):
        '''
        @intro：        ！！！不需要外部调用！！！
        @X:             输入X
        @Labels：       真实值
        @eplision：  保证传入MLEloss函数的输入的上下限
        @lossfunction： 指定损失函数，目前版本只支持：MLELoss(用于二分类)，RMSELoss（均方根误差：用于回归预测）
        '''

        if not self.trainMode or Labels is None:
            return Y
				# else：
        return dW_all, db_all, loss

      
  	
    def train(self, X, labels=None, valSet=None, valabel=None, lrate=0.01, epochs=10, batchSize=0, lossfunction="RMSELoss"):
        '''
        @X：        请确保输入的X不带有标签
        @labels：   输入的标签
        @valset：   验证集：    如果指定的话，则标签也应该给出，会在每10次迭代中对验证集进行预测和评估
        @valset：   验证集标签
        @lrate：    学习率
        @epochs：   迭代次数： 完整遍历一次训练集的次数
        @batchSize：批次大小：每次迭代的训练集大小；指定为0时，迭代大小为整个训练集大小（批梯度）
        @lossfunction： RMSELoss（均方根误差：回归），MLELoss（似然误差：二分类）
        @return：   Loss（每10次迭代进行一次记录），ac_t（训练集准确率，每10次迭代记录一次），ac_v(验证集准确率：每10次迭代记录一次)     
        '''
        
        # ensure X NOT append with label
    
        return Loss, ac_t, ac_v

    # return Y, X
    def Fullnet(self, X, W, b):
        return X.dot(W)+b, X

    '''
    #   带有backward的指出这个函数是用于计算反向传播
    '''

    # 预测
    def predict(self, datas:Iterable, labels=None, threshold=0.5):
        '''
        @data:      输入数据，请确保没有label
        @labels:    默认为None：仅输出神经网络的预测值；如果指定了labels，按照阈值将预测值转换为类别，并计算准确率    
        @threshold：阈值
        @return：   预测值（以及准确率，如果指定了labels的话）
        '''
        # fit
        self.eval()
        res = self.Loss(datas, labels)
        if labels is not None:
            res = (res > threshold).astype('int').reshape((-1, 1))
            return res, (res==labels).mean()
        else:
            return res
```

数据划分的主要函数：

```python
class DataLoader:
  	# 请保证datas中包含label
    def __init__(self, datas:Iterable) -> None:
        self.datas = datas

    #数据分解
    def KfolderData(self, K:int, shuffle=False, test=False):
        '''
        @K：        K折交叉验证的K值
        @shuffle：  划分验证集的时候是否打乱数据
        @test：     True时将只返回一组训练集和验证集（用于调试，不需要进行K次）
        @return：   返回生成器：可用for进行迭代，每次产出：训练集数据、训练集标签（已矫正为N*1）、验证集数据、验证集标签（已矫正为N*1）
        '''
            for i in range(1):
                yield trainset[:,:-1], trainset[:,-1].reshape(-1,1), valset[:,:-1], valset[:,-1].reshape(-1,1)
    
    
    
    # ！！不需要显式调用！！
    def Batch(self, batchsize=100, shuffle=True, throw=True):
        '''
        @batchsize: 批大小
        @shuffle：  是否打乱数据
        @throw：    是否丢弃最后一组（大小不满足batchsize）
        @return：   训练数据、标签（已矫正为N*1）
        '''
            yield self.datas[batchsize*i:batchsize*(i+1),:-1], self.datas[batchsize*i:batchsize*(i+1),-1].reshape(-1,1)
    
```



调用示例：

```python
# read file
#dataset = ... (numpy array)
dataset = pd.read_csv(file,header=None).values
# 将数据放入DataLoader，以便能直接进行训练集划分
loader = DataLoader(dataset)

# 进行K折
for trainset，trlabel, valset, valabel in loader.KfolderData(k=5, shuffle=True):
  	# 创建一个模型: 直接从文件加载参数，并自动识别layers信息
    model = NeuralNet("paramfile")
    # or 手动输入层数信息
		# 创建一个两层的网络：40*10， 10*1，默认激活函数为sigmod：
    #model = NerualNet(featureDim=40, outputDim=1, HiddenDims=[10])
    
    # train
    Loss = model.train(trainset, trlabel, lrate=0.1,epochs=100,batchsize=1000,lossfunction="RMSELoss")
    
    # 调整至非训练模式
    model.eval()
    # 预测
    pred = model.predict(valset)
    # 计算准确率：比较pred和valabel
    。。。
    # 保存这次训练的参数
    model.saveParam(file="paramfile")
```

