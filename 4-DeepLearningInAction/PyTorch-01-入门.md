
<h1 style=" text-align: center; font-size: 3em; font-family: 'Georgia', serif; color: #2c3e50; margin: 0.5em 0; padding: 10px 0; border-top: 4px solid #3498db; border-bottom: 4px solid #3498db; text-transform: uppercase; letter-spacing: 3px;">PyTorch入门</h1>


### 🎯 学习目标 ：
- 熟练掌握PyTorch张量常用操作
- 学会使用PyTorch自动求导
- 学会使用Dataset加载和预处理数据集
- 学会使用PyTorch解决**线性回归 和 逻辑回归**问题


### 神经网络和张量
![nn and tensor.png](../images/nn%20and%20tensor.png)

神经网络的输入和输出都是张量， 网络本身是一组参数组成，这些参数也是张量
![image as tensor.png](../images/image%20as%20tensor.png)

### 张量操作

更多PyTorch张量操作命令查阅 --> [机器学习工具最常用命令合集#PyTorch 张量操作](../1-EssentialBasics/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E5%85%B7%E6%9C%80%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4%E5%90%88%E9%9B%86.md#PyTorch 张量操作)

[PyTorch-入门-张量](https://github.com/jimmy-pink/colab-playground/blob/main/3pytorch/PyTorch-%E5%85%A5%E9%97%A8-%E5%BC%A0%E9%87%8F.ipynb)

### 自动求导

[PyTorch-入门-自动求导](https://github.com/jimmy-pink/colab-playground/blob/main/3pytorch/PyTorch-%E5%85%A5%E9%97%A8-%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC%E7%AD%89.ipynb)
- 自动求导
- 求偏导数
```python
import torch # 创建一个需要梯度的张量 
x = torch.tensor(3.0, requires_grad=True)  
y = x ** 2 # y = x^2 = 9  
# 计算梯度  
y.backward() # dy/dx = 2x = 6  
print(x.grad.tolist()) # 输出: tensor(6.) (因为 x=3, 2*3=6)
```

### Dataset
[PyTorch-入门-Dataset](https://github.com/jimmy-pink/colab-playground/blob/main/3pytorch/PyTorch-%E5%85%A5%E9%97%A8-Dataset.ipynb)

- 内置Dataset
- 自定义Dataset
- 数据transform，数据增强


### 线性回归

https://github.com/jimmy-pink/colab-playground/blob/main/3pytorch/PyTorch-%E5%85%A5%E9%97%A8-Regression.ipynb

- 简单线性回归
	- 公式：$y = wx + b$
		- x 是输入 （1个数）；
		- w 是权重（PyTorch 自动初始化）；
		- b 是偏置（bias，也会自动初始化）；
		- y 是输出（1个数）。
	- 定义线性回归模型
```python
class LR(nn.Module):

    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LR() 

X = torch.tensor([[1.0], [2.0], [3.0]])

# 定义损失函数 和 优化函数
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  
  
train_losses = []
# 模型训练
for epoch in range(10):  
    # --- 核心五步法 ---    optimizer.zero_grad()        # 1. 清零梯度 (擦黑板)  
    y_hat = model(X)             # 2. 前向传播 (算预测值)  
    loss = criterion(y_hat, Y)   # 3. 计算损失 (看错多少)  
    loss.backward()              # 4. 反向传播 (算导数/梯度)  
    optimizer.step()             # 5. 更新参数 (迈出一小步)  
  
    train_losses.append(loss.item()) # 6.保存Loss值  
    if (epoch+1) % 100 == 0:  
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
        
# 模型评估
model.eval()  
  
# 2. 准备测试集 (必须也是 2D Tensor)X_test = torch.tensor([[5.0], [10.0], [20.0]], dtype=torch.float32)  
  
# 3. 预测 (禁用梯度计算，这是预测时的标准姿势)  
with torch.no_grad():  
    predictions = model(X_test)  
  
# 4. 查看结果  
for i, x in enumerate(X_test):  
    print(f"输入 x = {x.item():.1f}, 预测 y_hat = {predictions[i].item():.4f} (理论值: {2*x.item()+3})")
```

- 多维线性回归 Multi Input Linear Regression
	- 公式：$y=w_0x_0+w_1x_1+w_2x_2+b$
	- **向量写法**： y = wx + b 
		- x 代表特征向量 Feature Vector
		- w 代表权重向量 Weight Vector ，$w[i]$ 和 $x[i]$ 一一对应，即x向量的第i个元素的权重参数是w向量的第i个元素
		- b 仍是标量
		- 特征向量与权重向量点乘 得到多维线性方程。
	- ![Pasted image 20260201155313.png](../images/Pasted%20image%2020260201155313.png)
	 [^1]: Image sourced from Coursera: "Introduction to NN and PyTorch" module 4 by IBM.

- 多输出线性回归 Multi Output Linear Regression
	- 公式：  y = Wx + b 
		- x 代表特征向量 Feature Vector ， 如，(x1,x2,x3)
		- W是特征矩阵， 注意这里是矩阵，因为每个y结果受一个权重向量$W[i]$的影响
		- y  代表多个目标值
		- b 是一个偏置向量
	- 举例：以体重秤为例，测量值x1为压力感应值，x2=生物电阻抗值，目标值y1=体重，y2=体脂 y3=血糖  ，则权重w1=$[w_\text{(0,0),} w_\text{(0,1)}],b_0$影响y1的计算, w2=$[w_\text{(1,0),} w_\text{(1,1)}], b_1$影响y2值，w3=$[w_\text{(2,0),} w_\text{(2,1)}], b_2$影响y3的计算。
- Linear Classifier
> 思考：线性回归如何解决二分类问题？优缺点是什么？
> 如：使用历年MPNP申请者数据：海外工作年限w1，雅思成绩w2, 居住时间w3,训练模型，训练线性回归模型，用于给新的申请人模拟打分，评估通过率。$$P = \text{Sigmoid}(w_1x_1 + w_2x_2 + w_3x_3 + b)$$
- [逻辑回归](2.5%20%E5%9B%9E%E5%BD%92%E5%92%8C%E5%88%86%E7%B1%BB#逻辑回归 (Logistic Regression))
	- 是 线性函数 + 激活函数 组成
	- 二分类使用Sigmoid作为激活函数，多分类使用softmax，也称为Logistic Function
	