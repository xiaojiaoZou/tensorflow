# 以下两行是消除一个warning用的
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# 基本概念
'''
tf.Tensor类的实例来表示张量，其实就是0到n维的数组的统称，每个 tf.Tensor有两个属性：
dtype Tensor 存储的数据的类型，可以为tf.float32、tf.int32、tf.string…
shape Tensor 存储的多维数组中每个维度的数组中元素的个数，如上面例子中的shape
在 TensorFlow 中，每个节点都是 tf.Tensor的一个实例
数据流是一种常用的并行计算编程模型，数据流图是由节点(nodes)和线(edges)构成的有向图：
节点(nodes) 表示计算单元，也可以是输入的起点或者输出的终点
线(edges) 表示节点之间的输入/输出关系
tensor种类：
constant()函数提供在tensorflow中定义常量(不可更改的张量)的方法
Variable()定义变量，TensorFlow中的变量特指深度学习机制中，控制输入到输出映射的可以变化的数据，这些变化数据随着训练迭代的进行，不断地改变数值，不断优化，使输出的结果越来越接近于正确的结果
placeholder()又叫占位符，用于声明一个张量的数据格式，告诉系统这里会有一个这种格式的张量，但是还没有给定具体数值，具体的数值要在正式运行的时候给到。占位变量是一种TensorFlow用来解决读取大量训练数据问题的机制，它允许你现在不用给它赋值，随着训练的开始，再把训练数据传送给训练网络学习。

tf的session
python做计算时numpy，需要计算时用C++，不需要计算时又切换回python，这样语言之间切换，造成耗时。
tf是先用Python、Java等代码用来设计、定义模型，构建的Graph，但不计算，构建完成后用tf.Session.run()方法传递给底层执行计算
session.run返回的张量全是numpy数组
'''

#基本操作
'''
# 创建常量
hello = tf.constant('Hello, TensorFlow!')
t0 = tf.constant(3, dtype=tf.int32)
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])
# 创建变量
x = tf.Variable(5, name='x') #有个trainable参数表示此变量的值是否能被像tf.optimizer修改，默认True，可以被训练
# 创建占位符
p1 = tf.placeholder(dtype=tf.int32)
print(hello)
print(x)
print(p1)
# print 一个 Tensor 只能打印出它的属性定义，并不能打印出它的值，（因为此时还没进行计算），要想查看一个 Tensor 中的值还需要经过Session 运行一下：
sess = tf.Session()
print(sess.run(t0))
sess.close()

#数据类型转换
tf.cast(x, tf.float32)

# Tensor 即可以表示输入、输出的端点，还可以表示计算单元
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
sess = tf.Session()
print(sess.run(adder))

# 创建两个占位 Tensor 节点
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder_node = a + b
# 打印三个节点
print(a)
print(b)
print(adder)
# 运行一下，后面的 dict 参数是为占位 Tensor 提供输入数据
sess = tf.Session()
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 添加×操作
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

a=tf.constant([1,2,3,4])
b=tf.reduce_prod(a) #沿某个维度连乘，默认所有维度
c=tf.reduce_sum(a) #沿某个维度连加，默认所有维度

#session模式
#第一种
sess = tf.Session()
sess.run(c)
sess.close()
#第一种模式在程序因异常退出时，关闭会话的函数可能不被执行从而导致资源泄漏
#第二种
with tf.Session() as sess:
    sess.run()
#第二种模式是通过上下文管理器来管理，当上下文退出时会话关闭和资源释放也自动完成

#tf.InteractiveSession()
sess=tf.InteractiveSession() #相当于创建了一个session同时还吧这个session设置成默认session了，所以eval操作执行的时候，不必指定session，因为有默认的了
print (c.eval())
#如果只是：
sess=tf.Session()
print (c.eval(session=sess))

#run的feed_dict参数给tensor对象赋值
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
sess=tf.Session()
sess.run(a+b,feed_dict={a:[1,2],b:[3,4]})

# 初始化变量，给session追踪，不同session是分别管理变量的，就是说同一个变量，在不同的session中值可能不同
init = tf.global_variables_initializer() #initialize_all_variables已弃用
sess = tf.Session()
sess.run(init)

#变量的自加 自减
a.assign_add(123)
a.assign_sub(123)

#名称作用域(name scope)组织数据流图，将Op划分到较大有名称语句块。TensorBoard加载数据流图，名称作用域封装Op。可以把名称作用域嵌在其他名称作用域内
#就是把图分成一块一块好管理
with tf.name_scope('scope_a'):
    a = tf.add(1,2, name = 'a')
    b = tf.multiply(3, 4, name = 'b')

#tf.Graph   
#你一旦开始你的任务，就已经有一个默认的图已经创建好了。而且可以通过调用tf.get_default_graph()来访问到
print(tf.get_default_graph()) #获取自动创建的图
c=tf.constant(value=1)
print(c.graph)  #获取c所在的图，就是上面那个默认的图

g=tf.Graph() #创建了一个新图
print("g:",g)
with g.as_default(): #以下语句创建的tensor和op都将在g中
    d=tf.constant(value=2)
    print(d.graph)

sess = tf.Session(graph = g) #指定此session要运行哪个图，默认是自动创建的默认图

#有用的函数
tf.sigmoid(x)
tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y) #计算交叉熵，logit回归损失函数
tf.one_hot(labels,depth) #把标签列表转化成矩阵形式，每行是一个样本，每列是类别编号，一个样本属于哪个类别，哪一列就是1，其余是0，depth指定总类别数

#TensorFlow高层封装
#tensorflow.contrib.slim它既不如原生态TensorFlow的灵活，也不如下面将要介绍的其他高层封装简洁
#tf.contrib.learn可以和使用sklearn类似的方法
#TFLearn需要单独安装,进一步简化了tf.contrib.learn中对模型定义的方法，并提供了一些更加简洁的方法来定义神经网络的结构
#Keras基于TensorFlow或者Theano的高层API，在安装好TensorFlow之后可以安装

tf.nn.conv2d(input,filter,strides,padding) 
#对4Darray进行卷积，4个维度包括[batch, height, width, channels]样本、图高度、图宽度、通道（rgb）
tf.nn.bias_add(value,bias)
#bias必须是1D，用广播的原则把bias加到value上
tf.nn.relu(x)
#relu激活函数，即把小于0的值归为0




'''

# 创建线性模型y=W×x+b
'''
# 样本数据如下：
# 1	4.8
# 2	8.5
# 3	10.4
# 6	21
# 8	25.3
# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W*x + b

# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))

# 通过tf.Variable()创建变量 Tensor 时需要设置一个初始值，但这个初始值并不能立即使用，变量 Tensor 需要经过下面的 init 过程后才能使用：
# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#运行一下模型
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
#运行一下损失函数
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# 用tf.assign给 W 和 b 赋新值
fixW = tf.assign(W, [2.])
fixb = tf.assign(b, [1.])
# run 之后新值才会生效
sess.run([fixW, fixb])
# 重新验证损失值
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
# 训练10000次
for i in range(10000):
    sess.run(train, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]})
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]})))
'''

# 用tf.estimator构建一个线性回归模型
'''
# tf.estimator是TensorFlow提供的高级库，提供了很多常用的训练模型

# 创建一个特征向量
f1=tf.feature_column.numeric_column("x1", shape=[1])
f2=tf.feature_column.numeric_column("x2", shape=[1])
feature_columns = [f1,f2]
# 创建一个LinearRegressor训练器，并传入特征向量列表
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 创建一个输入模型，用来训练
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x1": np.array([1.,2.,3.]),"x2":np.array([4.,5.,6.])}, np.array([5.,7.,9.]), batch_size=2, num_epochs=None, shuffle=True)
# 使用训练数据训练1000次
estimator.train(input_fn=train_input_fn, steps=1000)

# 再用训练数据创建一个输入模型，用来进行后面的模型评估
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x1": np.array([1.,2.,3.]),"x2":np.array([4.,5.,6.])}, np.array([5.,7.,9.]), batch_size=2, num_epochs=1000, shuffle=False)
# 使用原来训练数据评估一下模型，目的是查看训练的结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)

# 再用测试数据创建一个输入模型，用来进行模型评估
train_input_fn_3 = tf.estimator.inputs.numpy_input_fn(
    {"x1": np.array([7.,8.,9.]),"x2":np.array([10.,11.,12.])}, np.array([17.,19.,21.]), batch_size=2, num_epochs=1000, shuffle=False)
# 使用原来训练数据评估一下模型，目的是查看训练的结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_3)
print("test metrics: %r" % train_metrics)
'''

# 自定义Estimator模型
'''
#如下定义一个函数，赋给tf.estimator.Estimator的model_fn参数即可实现定义自己的模型函数、损失函数、训练方法
def model_fn(features, labels, mode):
    # 构建线性模型
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x']+ b
    # 构建损失模型
    loss = tf.reduce_sum(tf.square(y - labels))
    # 训练模型子图
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    # 通过EstimatorSpec指定我们的训练子图积极损失模型
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)
estimator = tf.estimator.Estimator(model_fn=model_fn)

x_train = np.array([1., 2., 3., 4., 5.])
y_train = np.array([7.,9.,11.,13.,15.])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)
estimator.train(input_fn=train_input_fn, steps=1000)

train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)
for n in estimator.get_variable_names():
    print(estimator.get_variable_value(n))
'''

# 实现神经网络
'''
# 定义训练数据 batch 的大小
from numpy.random import RandomState
batch_size = 8 #每次最优化取8个样本

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.placeholder(tf.float32 ,shape = (None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1) ,name = 'y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) #使用交叉熵损失函数
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) #使用adma优化方法

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = np.array([[int(x1 + x2 < 1) for (x1, x2) in X]]).reshape(128, 1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)

    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练轮数
    STEPS = 1000
    for i in range(STEPS):

        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # print("%d,%d" % (start,end))
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict = {x : X[start : end], y_ : Y[start : end]})

        if i % 100 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict = {x : X, y_ : Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
'''

# 实现逻辑回归 自练习
'''
a=tf.constant(1,dtype=tf.float64,name='a')
w=tf.Variable(np.array([[1],[2],[3]]),dtype=tf.float64,name='w')
b=tf.Variable(0.5,dtype=tf.float64,name='b')
x=tf.placeholder(shape=[None,3],dtype=tf.float64,name='x')
y=tf.placeholder(shape=[None,1],dtype=tf.float64,name='y')

z=tf.matmul(x,w)+b
model=a/(a+tf.exp(-z))
loss=-tf.reduce_sum(y*tf.log(model)+(a-y)*tf.log(a-model))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

x_input=np.array([[1,0,0],
                  [0,0,1],
                  [1,1,0],
                  [0,1,1]],dtype=np.float64)
y_input=np.array([[1],[0],[1],[0]])

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train,feed_dict={x:x_input,y:y_input})
    w_,b_,model_,loss_=sess.run([w,b,model,loss],feed_dict={x:x_input,y:y_input})
    print(w_)
    print(b_)
    print(model_)
    print(loss_)
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

