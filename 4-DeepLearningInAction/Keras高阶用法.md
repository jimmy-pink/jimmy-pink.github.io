

## **一、理解 Keras 的灵活性边界**

Keras 提供三种主要的开发模式：

- **函数式 API（Functional API）**：用于快速搭建标准神经网络结构。
    
- **子类化模型（Model Subclassing）**：允许你自由定义模型结构和前向传播逻辑。
    
- **自定义训练循环（Custom Training Loops）**：你可以完全控制前向传播、反向传播、优化和日志记录等流程。
    

高阶用法主要围绕子类化与自定义训练展开。

---

## **二、自定义训练流程：掌控每一步**

  

当你需要更精细地控制训练过程（如使用非标准 loss，加入特定监控逻辑等），可以使用 GradientTape 手动编写训练循环，或在 Model 子类中重写 train_step 方法。


**优点**：
- 自定义Loss函数和优化方法
- 高级日志和监控
-  研究与开发的灵活性
- 集成自定义操作和模型层 

**常见场景包括：**

- 多任务学习：同时训练多个输出和 loss。
    
- 动态 loss 加权：根据 epoch 调整损失函数比重。
    
- 强化学习或 GAN：需对训练步骤进行定制控制。
    

- 🌰 使用GradientTape的自定义训练
```python
import tensorflow as tf

model = MyModel()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape: #记录前向传播中的运算
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

-  🌰 自定义训练方法：重写Model的train_step
适用于model.fit()框架下需要更灵活逻辑的情况：
```python
class MyModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```


---

## **三、设计自定义 Layer 与模型组件**

  

对于复杂网络结构、稀有操作或代码复用需求，Keras 支持通过继承 Layer 或 Model 类创建自定义模块。

- 🌰 自定义Layer
```python
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
- 🌰自定义模型结构 (子类化Model)
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```


### **3.1 自定义 Layer 的典型用途**

- 实现新颖的前向传播逻辑（如 capsule、attention 等）
    
- 管理权重的初始化与保存
    
- 添加自定义 loss 或 metric
    

  

### **3.2 自定义 Model 的常见场景**

- 多分支模型（如 Siamese 网络）
    
- 输出结构复杂、无法用 Sequential 或 Functional 表达
    
- 内嵌多个自定义子模型

---

## **四、高级回调函数：训练过程的控制中心**

  

Keras 的 Callback 系统提供训练中途干预、日志记录、保存模型、动态调整超参数等功能。
  

**常用回调包括：**

| **回调名**               | **功能描述**      |
| --------------------- | ------------- |
| ModelCheckpoint       | 保存最优模型或周期性检查点 |
| EarlyStopping         | 自动停止训练，防止过拟合  |
| ReduceLROnPlateau     | 监控指标无提升时降低学习率 |
| TensorBoard           | 记录训练过程，支持可视化  |
| LearningRateScheduler | 动态调节学习率策略     |
高级用户还可以通过继承 Callback 编写自定义回调，用于日志上传、模型精度评估等功能。

- 🌰 自定义Callback
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] > 0.9:
            print("Validation accuracy over 90%, stopping training.")
            self.model.stop_training = True
```

---

## **五、混合精度与分布式训练**

  

Keras 支持通过以下手段提升训练效率，尤其适用于大型模型或多 GPU 环境：

  

### **5.1 混合精度训练（Mixed Precision）**

  

启用后，模型将自动在保持数值稳定性的前提下使用半精度浮点数（float16）进行加速，显著提升性能。

- 🌰 混合精度训练
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### **5.2 分布式训练策略**

  - 🌰 多GPU分布式训练
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
    model.compile(optimizer, loss, metrics)
```

TensorFlow 的分布式策略（如 MirroredStrategy）可用于多卡同步训练，适合大规模数据或模型。

---

## **六、构建自定义 Loss 与 Metric**

  

在一些特定任务中，Keras 默认的损失函数和指标可能无法满足需求，例如：

- 异构任务的自定义损失组合
    
- 多标签分类下的自定义 F1-score
    
- 特定业务需求下的加权误差惩罚

你可以继承 Loss 或 Metric 类，封装逻辑并与 model.compile() 配合使用，享受标准训练框架带来的易用性。

- 🌰 自定义Loss和Metric
```python
class MyCustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

class MyMetric(tf.keras.metrics.Metric):
    def __init__(self, name="my_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(tf.abs(y_true - y_pred)))
```

---

## **七、进阶技巧补充**

- add_metric() 和 add_loss()：可在 Layer 中注入训练指标和额外损失。
    
- 模型微调（Fine-tuning）：冻结部分层，组合训练策略。
    
- 模块复用：将多个自定义 Layer 打包为子模块，便于在不同项目中迁移。
    


- 🌰 add_metric, add_loss 自定义模型中嵌入指标与损失
```python 
def call(self, inputs):
    output = self.dense(inputs)
    self.add_metric(tf.reduce_mean(output), name='output_mean')
    self.add_loss(tf.reduce_sum(output))
    return output
```


---