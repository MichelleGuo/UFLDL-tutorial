###Linear regression

Main idea:熟悉目标函数，计算梯度，并在一系列参数上对目标函数进行优化
CS229d-note1 监督学习for more  
Our goal:线性回归目标--从输入向量$x\in\Re^n$预测目标值$y$  
Example: 预测房价的例子中，$y$表示房价，$x$向量中的元素$x_j$表示房子的特征（features），比如面积、房间数等等。假设给定许多实例来表示这种关系：实例集合中第$i$个房屋的特征向量为$x^{(i)}$，价格为$y^{(i)}$。简单来说目标就是找到$y = h(x)$使得训练集合中的每个实例都有：$y^{(i)} \approx h(x^{(i)})$。我们希望$h(x)$是很好的预测函数，即使在我们只给定一个新的房屋的特征（features）时也能有对其价格的准确预测。  
为了找到满足$y^{(i)} \approx h(x^{(i)})$的$y = h(x)$，需要决定$h(x)$的表达。  
首先我们使用线性方程：$h_\theta(x) = \sum_j \theta_j x_j = \theta^\top x$。$h_\theta(x)$表示在参数为$\theta$是的函数集合，可以叫做"hypothesis class"。那么任务就是找到适合的$theta$使得$h_\theta(x^{(i)})$尽可能接近$y^{(i)}$。即最小化损失函数：  
$J(\theta) = \frac{1}{2} \sum_i \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2} \sum_i \left( \theta^\top x^{(i)} - y^{(i)} \right)^2$  
This function is the “cost function” for our problem which measures how much error is incurred in predicting $y^{(i)}$ for a particular choice of $\theta$. 

####Function Minimization  
想要找到使得$J(\theta)$最小的$\theta$值，有许多算法可以实现，在之后的梯度下降会详细介绍。现在我们先假定为了最小化代价函数我们需要两个信息：计算$J(\theta)$和$\nabla_\theta J(\theta)$，剩下的过程就交给优化算法。  
$\nabla_\theta J(\theta)$是可导函数$J$的向量，方向为$J$的最速上升方向。所以优化算法利用其来增加或减小$J(\theta)$，而$\theta$的改变很小。

给定训练集$x^{(i)}$，$y^{(i)}$来计算$J(\theta)$，剩下只需考虑梯度的计算：
$\nabla_\theta J(\theta) = \begin{align}\left[\begin{array}{c} \frac{\partial J(\theta)}{\partial \theta_1}  \\
\frac{\partial J(\theta)}{\partial \theta_2}  \\
\vdots\\
\frac{\partial J(\theta)}{\partial \theta_n} \end{array}\right]\end{align}$  
对某个特定的参数$\theta_j$:$\frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j \left(h_\theta(x^{(i)}) - y^{(i)}\right)$


####Logistic Regression 
上一节学习了如何用输入值（房子面积）的线性函数来预测连续值（房价）。有时我们希望预测离散数值，比如预测某个像素点网格内代表“0”或者是“1”。这是分类问题。逻辑回归是用于做决策的一个简单分类算法。

在线性回归中，我们尝试用线性函数$y = h_\theta(x) = \theta^\top x.$去预测第$i$个实例$x^{(i)}$的值$y^{(i)}$,但对于预测二元值标签$\left(y^{(i)} \in \{0,1\}\right)$这并不是一个好的方法。
在Logistic Regression中，使用不同的假设类来预测给定实例$i$属于“0”或“1”类的概率：  
$ P(y=1|x) = h_\theta(x) = \frac{1}{1 + \exp(-\theta^\top x)}=\sigma(\theta^\top x),\\
P(y=0|x) = 1 - P(y=1|x) = 1 - h_\theta(x).$  

$\sigma(z) = \frac{1}{1+\exp(-z)}$为“sigmoid”或“logistic”函数，是个“S”型函数，将$\theta^\top x$的值映射到$[0,1]$中，因此我们可以将$h_\theta(x)$解释为概率。我们的目标是找到一个$\theta$使得概率$P(y=1|x) = h_\theta(x)$当$x$属于类“1”时很大，当$x$属于类“0”时概率很小（即$P(y=1|x)$很大）。对于有二元标签的训练集$\{ (x^{(i)}, y^{(i)}) : i=1,\ldots,m\}$，用以下损失函数来衡量$h_\theta$：  
$J(\theta) = - \sum_i \left(y^{(i)} \log( h_\theta(x^{(i)}) ) + (1 - y^{(i)}) \log( 1 - h_\theta(x^{(i)}) ) \right).$  
上式中两个加项只有一个为非零项，取决于$y^{(i)}$为0或者1。当$y^{(i)}=1$，需要最大化$h_\theta(x^{(i)})$，反之$y^{(i)}=0$时需要最小化$h_\theta$。
关于回归的解释和损失函数的推导详见CS229-note1。  
现在有了损失函数用来衡量假设函数$h_\theta$如何拟合训练数据，目标即找到使得$J(\theta$最小的参数$\theta$。对于新的某个unseen的测试点，为它分为类“1”的判断依据即$P(y=1|x) > P(y=0|x)$，反之则为“0”。同理即是判断是否$h_\theta(x) > 0.5$。  
最小化$J(\theta)$的方法和线性回归类似。对于任意的参数$\theta$需要计算$J(\theta$和$\nabla_\theta J(\theta)$。偏导数如下：
$\frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j (h_\theta(x^{(i)}) - y^{(i)}).$  
写成向量形式，整体梯度可以表示为：
$\nabla_\theta J(\theta) = \sum_i x^{(i)} (h_\theta(x^{(i)}) - y^{(i)})$


####Vectorization 
对于一些的小的学习任务，比如前面用于线性回归的房屋价格数据，并不需要非常快的运行速度。然而，使用loop循环来遍历对于大数据量的处理会很慢，MATLAB中对样本或者条目（elements）的顺序遍历是很耗时的，因此为了避免loop的使用，建议使用向量和矩阵的操作进行优化，加快运行速度。（在其他的编程语言中同样适用，包括Python,C/C++等，我们希望尽可能地复用优化操作）

#####Example: Many matrix-vector products 
我们经常需要计算矩阵-向量的乘积，比如我们需要计算数据集中每个样本的$\theta^\top x^{(i)}$，$\theta$可能是二维矩阵或者是向量，我们可以构造一个矩阵$X$来联结所有的样本$x^{(i)}$，使得其包含整个数据集。 
$X = \left[\begin{array}{cccc}
  | & |  &  | & | \\
  x^{(1)} & x^{(2)} & \cdots & x^{(m)}\\
    | & |  &  | & |\end{array}\right]$。  

我们可以一次性对所有的$x^{(i)}$计算$y^{(i)} = W x^{(i)}$：
$\left[\begin{array}{cccc}
| & |  &  | & | \\
y^{(1)} & y^{(2)} & \cdots & y^{(m)}\\
| & |  &  | & |\end{array}\right] = Y = W X$  
可以避免在使用线性回归时，计算过程中的循环遍历$y^{(i)}=\theta^\top X^{(i)}$。


####Softmax  
Softmax Regression（也称多项式逻辑回归：multinomial logistic regression）是logistic回归的泛化，针对的是多类问题。在logistic回归中，标签是二元的：$y^{(i)} \in \{0,1\}$，我们使用这样的分类器来区分两类手写数字，Softmax允许我们处理$y^{(i)} \in \{1,\ldots,K\}$的问题，$K$为类的数量。  
在Logistic回归中，使用$m$个标注样本的训练集$\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$，输入的特征为$x^{(i)} \in \Re^{n}$，在logistic回归中，二分类的标签$y^{(i)} \in \{0,1\}$，有如下假设函数：
$\begin{align} h_\theta(x) = \frac{1}{1+\exp(-\theta^\top x)}, \end{align}$
模型参数$\theta$可以通过训练来最小化如下损失函数：
\begin{align}
J(\theta) = -\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}


在Softmax回归中，与二元分类相对的是，我们更关注多元分类问题，即标签$y$可以有不同的值。还是使用$m$个标注样本的训练集$\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$，此时标签$y^{(i)} \in \{1, 2, \ldots, K\}$ （注意此处我们的index从1开始而非0）。比如在手写数字识别的任务中，$K=10$个不同的类别。

给定测试输入$x$，希望我们的假设函数可以计算出不同$k=1,2,...,K$下的概率值$P(y=k | x)$，因此，假设函数的输出是一个$K$维向量，向量内所有值相加和为1，$K$个概率值。假设函数有以下形式：
$\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\
\exp(\theta^{(2)\top} x ) \\
\vdots \\
\exp(\theta^{(K)\top} x ) \\
\end{bmatrix}
\end{align}$


$\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)} \in \Re^{n}$是模型的参数，注意到$\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) } }$对分布进行归一化，所以其和为1。

为了方便，我们仍用$\theta$表示模型所有的参数，当对Softmax进行实现时，通常将$\theta$表示为$n*K$的矩阵，将$\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)}$连接作为矩阵的列：
$\theta = \left[\begin{array}{cccc}| & | & | & | \\
\theta^{(1)} & \theta^{(2)} & \cdots & \theta^{(K)} \\
| & | & | & |
\end{array}\right].$

###Cost Function

损失函数如下：
$1\{\cdot\}$为指示函数，$1\{\hbox{a true statement}\}=1$，$1\{\hbox{a false statement}\}=0$，例如，$1\{2+2=4\}$ 值为1，$1\{1+1=5\}$值为0。
$\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}$
Logistic回归损失函数为：$\begin{align}
J(\theta) &= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}$
除了需要对K个不同分类的类标进行求和以外，Softmax的损失函数是相似的：
$P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }$

对于$J(\theta)$的最小化问题，目前还没有闭式解法。因此，我们使用迭代的优化算法（例如梯度下降法，或 L-BFGS）。经过求导，我们得到梯度公式如下：
$\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}$
当K=2时，可以看出Softmax和Logistic回归的关系：
$\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}
\end{align}$


