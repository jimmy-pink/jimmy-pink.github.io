
[Coursera课程地址](https://www.coursera.org/learn/introduction-tensorflow/home/week/4)

[Tensorflow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras/)

[GITEE python代码](https://gitee.com/golden-mountain/py-machine-learning)

[主讲老师劳伦斯的个人网站](https://laurencemoroney.com/about.html)

[Kaggle:机器学习赛事和挑战](https://www.kaggle.com/)

[Github Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist)

[Corsera Official Colab](https://www.coursera.org/learn/introduction-tensorflow/ungradedLab/6Hb8q/get-hands-on-with-computer-vision-lab-1/lab?path=%2Flab%2Ftree%2Flab_1)







##  计算机视觉入门实例
![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743341785507-a93abb14-20cd-40ae-8d6b-fcbd58901fbe.png)![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743341807155-d3d2158b-5fcd-4e06-afab-5ea430391476.png)

![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743341915667-6322b260-9587-469f-ba82-b27687a3e884.png)
### 浅层学习
#### <font style="background-color:#F6E1AC;color:black">快速上手</font>训练一批数据，使其掌握 y=2x+1 的规律
`x_array = [0, 1, 2, 3, 4, 5]` 特征  
`y_array = [-1, 1, 3, 5, 7, 9]` 标签

经过 n 轮训练，计算机将掌握 y=2+1 的规律，给出一些测试值$x_i$，模型将给出$y_i = 2* x_i +1$的近似值
```python
import tensorflow as tf
import numpy as np

# 1. 构造训练数据 (y = 2x - 1)
def create_training_data():

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms.
    # For this exercise, please arrange the values in ascending order (i.e. 1, 2, 3, and so on).
    x_train = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)  # 输入数据
    y_train = np.array([-1, 1, 3, 5, 7, 9], dtype=np.float32)   # 真实标签, 其实只有6个数[-1, 1, 3, 5, 7, 9]， x_train的值只有六个


    return x_train, x_train

def define_and_compile_model():

    # Define a compiled (but untrained) model。 定义模型，就是定义模型的神经图络层。
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),  # 输入层：1个输入特征
        tf.keras.layers.Dense(units=1)  # 输出层：1个神经元
    ])

    # 编译模型
    # sgd（Stochastic Gradient Descent，随机梯度下降）
    # mse（Mean Squared Error，均方误差），衡量预测值与真实值之间的差距。
    model.compile(optimizer='sgd', loss="mse")

    return model

def train_model():

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember you already coded a function that does this!
    x_train, y_train = create_training_data()

    model = define_and_compile_model()

    # Train your model for 500 epochs by feeding the training data
    model.fit(x_train, y_train, epochs=500, verbose=0)

    return model


# 保存模型
model = train_model()
model.save('my_model.keras')

# 加载模型
loaded_model = tf.keras.models.load_model('my_model.keras')

# 使用模型进行预测
xn = np.array([6], dtype=np.float32)  # 需要预测的 x 值
yn_pred = loaded_model.predict(xn)

print(f'输入 xn={xn[0]}，预测输出 yn={yn_pred}')

```

#### 收敛 Convergence
<font style="color:rgba(0, 0, 0, 0.87);">convergence 收敛 (接近于)， refers to the process of getting closer to the correct answer.</font>

+ 损失函数收敛 （loss convergence）
+ 预测结果收敛   (prediction convergence)

#### 特征 Features  和标签 Labels
输入的数据 称为**特征** Features 或输入变量

告诉电脑输入的数据代表什么，或结果，称为**标签** label、**目标** target 或输出变量

#### Loss / Optimizor
loss 函数 用于衡量 模型预测值 与 实际值 之间的差异。

optimizor 调整模型权重 以最小化 loss 值

###  计算机视觉入门
Computer vision is the field of having a computer understand and label what is present in an image.

#### <font style="background:#F6E1AC;color:#664900">快速上手</font>Fashion Mnist 训练模型识别服装类型
### 使用卷积增强视觉效果
#### 卷积层 
为图像加滤镜，以突出特征

`tf.keras.layers.Conv2D(16, (3,3), activation='relu')`

在输入特征图上滑动 **16 **个** **3×3 的**滤波器（卷积核），**通过** 卷积计算 **提取局部特征。

结果生成 16 个新的特征图

激活函数 relu： 会对负值进行截断（设为 0），保持非线性特性。

#### 汇集层 **Pooling Layer**
汇集层也叫**池化层，**有最大池，平均池，全局平均池。

**汇集层**是 **卷积神经网络**中用于**降低数据维度、减少计算量、提高特征稳定性**的一种操作。

作用： 

1. **降低计算量**：减少特征图的尺寸，减少后续计算。
2. **防止过拟合**：减少参数，提高模型的泛化能力。
3. **增强特征稳定性**：即使输入图片有小幅度变化（如平移、旋转），汇集层能保持特征不变。

`tf.keras.layers.MaxPooling2D(2,2)`

**最大池化**的作用是对输入特征图（Feature Map）分块，并在每个 2×2 块中取最大值，生成新的较小特征图。

取 2*2 像素的最大值

#### 二分类问题

+ 二分类问题： 如垃圾邮件检测、疾病诊断（有/无）、图像分类（如猫/狗）等。
+ Binary Crossentropy 二分类问题使用该loss函数
+ RMSProp 一种自适应学习率的优化算法，通过调整每个参数的学习率来加速训练，减少震荡
+ Sigmoid 用于输出层将值压缩到0-1之间，以表示二分类的类别归属。

#### 过滤拟合
模型记住了训练数据中的特定模式，但未能很好地<font style="color:rgba(0, 0, 0, 0.87);color:black">泛化</font>到验证数据。 

比如在训练识别人和马的模型时，如果全部使用站立的人训练， 在验证时使用坐着的人像时可能导致预测错误， 这时就说明训练数据过度拟合了。
## 卷积神经网络 CNN

### 数据清洗
+ 缺失值处理： <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据集中某些特征缺失</font>
+ 重复数据去除： <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据集中存在完全相同的记录</font>
+ 异常值检测： <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据中存在明显偏离正常范围的值</font>
+ 数据一致性：  <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据格式不统一，如日期格式不同</font>
+ 数据标准化与归一化： <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据分布范围不均匀</font>
+ 噪声数据处理： <font style="color:rgba(0, 0, 0, 0.87);background-color:rgb(225, 245, 254);">数据中包含无效或无意义的信息</font>

### 数据增强 Data Augment
在不增加数据规模的情况下，通过对数据增强，来减少训练数据的过滤拟合。

对原始数据进行一定的变换 和  合成， 提高训练模型的健壮性。

数据增加很花费 CPU/GPU，可能会导致训练很慢。

![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743345449634-4607cae8-1123-40a8-bac0-91686938d171.png)

![增加：旋转45度](https://cdn.nlark.com/yuque/0/2025/png/295096/1743345472152-a2d86b0f-4d23-4f3d-aa7d-293fcc439471.png)![增强：水平翻转](https://cdn.nlark.com/yuque/0/2025/png/295096/1743345500033-cc84cff8-df8f-4369-8a6f-87436074ec21.png)![增强：局部缩放](https://cdn.nlark.com/yuque/0/2025/png/295096/1743345569972-f1da3dfa-97a0-409d-9775-3ac96707ee1f.png)

### 迁移学习 Transfor Learning
基于一个已存在的模型，训练出一个新模型。

#### 多级分类问题

`tf.keras.layers.Dense(3, activation='softmax')`

`model_with_aug.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',)`

## NLP


